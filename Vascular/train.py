# nohup env CUDA_VISIBLE_DEVICES=2 python -u train.py > train_log.txt 2>&1 &

import numpy as np
from dataset.vascular_data_utils_multi_label import get_loader_vascular
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import os
import wandb

set_determinism(123)

data_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/imagesTr"
logdir = "./logs_vascular/diffusion_seg/"
model_save_path = os.path.join(logdir, "model")
os.makedirs(model_save_path, exist_ok=True)

env = "pytorch" 
max_epoch = 1000
batch_size = 4
val_every = 10
num_gpus = 1
device = "cuda:0"

number_modality = 2
number_targets = 5

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class VascularTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        wandb.init(project="Vascular_Diffusion_Seg", 
                   name="diffusion_training", 
                   settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                   config={
                        "max_epochs": max_epochs,
                        "batch_size": batch_size,
                        "lr": 1e-4,
                        "num_targets": number_targets
                   })

        wandb.define_metric("Validation/*", step_metric="global_step")
        wandb.define_metric("Training/*", step_metric="global_step")

        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.25)
        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                    warmup_epochs=30,
                                                    max_epochs=max_epochs)
        self.ce = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(softmax=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        x_start = (label * 2) - 1
        
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        pred_01 = (pred_xstart + 1) / 2

        loss_dice = self.dice_loss(pred_01, label)
        
        target_indices = torch.argmax(label, dim=1).long()
        loss_ce = self.ce(pred_01, target_indices)
        
        loss_mse = self.mse(pred_01, label)

        loss = loss_dice + loss_ce + loss_mse
        
        with torch.no_grad():
            train_dice_score = 1.0 - loss_dice.item()

        self.log("Training/total_loss", loss, step=self.global_step)
        wandb.log({
            "Training/total_loss": loss.item(),
            "Training/dice_loss": loss_dice.item(),
            "Training/dice_score_overall": train_dice_score,
            "Training/ce_loss": loss_ce.item(),
            "Training/mse_loss": loss_mse.item(),
            "Training/lr": self.optimizer.param_groups[0]['lr'],
            "global_step": self.global_step
        })
        
        return loss
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"].float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output_final = torch.argmax(output, dim=1)
        target_final = torch.argmax(label, dim=1)
        
        output_np = output_final.cpu().numpy()
        target_np = target_final.cpu().numpy()
        
        results = []
        for i in range(1,5):
            results.append(dice(output_np == i, target_np == i))

        return results

    def validation_end(self, mean_val_outputs):
        d1, d2, d3, d4 = mean_val_outputs
        mean_dice = sum(mean_val_outputs) / len(mean_val_outputs)

        log_dict = {
            "Validation/dice_abdominal": d1,
            "Validation/dice_aortic_arch": d2,
            "Validation/dice_ascending": d3,
            "Validation/dice_descending": d4,
            "Validation/mean_dice_score": mean_dice,
            "Validation/epoch": self.epoch,
            "global_step": self.global_step
        }

        self.log("mean_dice", mean_dice, step=self.epoch)
        wandb.log(log_dict)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"Mean Dice Score: {mean_dice:.4f} (Abdom: {d1:.4f}, Arch: {d2:.4f}, Asc: {d3:.4f}, Desc: {d4:.4f})")

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_loader_vascular(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = VascularTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=18899,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
