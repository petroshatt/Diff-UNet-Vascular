# nohup env CUDA_VISIBLE_DEVICES=2 python -u train.py > train_log.txt 2>&1 &

import numpy as np
from dataset.vascular_data_utils_multi_label import get_loader_vascular
from dataset.vascular_data_utils_single_modality import get_loader_vascular_single_modality
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
from utils.curriculum import to_binary_onehot, pred5_to_pred2
import os
import wandb

set_determinism(123)

data_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/imagesTr"
logdir = "./logs_vascular/diffusion_seg/"
model_save_path = os.path.join(logdir, "model")
os.makedirs(model_save_path, exist_ok=True)

env = "pytorch" 
max_epoch = 1000
batch_size = 2
val_every = 20
num_gpus = 1
device = "cuda:0"

number_modality = 2
number_targets = 5

curriculum_pretrain_epochs = 180

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
        
        wandb.init(project="Diff-UNet-Curriculum-Vascular", 
                   name=f"Diff-UNet Curriculum (Binary->{number_targets-1} regions)", 
                   settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                   config={
                        "max_epochs": max_epochs,
                        "curriculum_pretrain_epochs": curriculum_pretrain_epochs,
                        "batch_size": batch_size,
                        "lr": 1e-4,
                        "num_targets": number_targets,
                        "ds_weights": [0.5, 0.25, 0.125, 0.0625]
                   })

        wandb.define_metric("Validation/*", step_metric="global_step")
        wandb.define_metric("Training/*", step_metric="global_step")

        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                                 sw_batch_size=1,
                                                 overlap=0.25)

        self.model = DiffUNet()

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=1e-4,
                                           weight_decay=1e-3)

        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=30,
                                                       max_epochs=max_epochs)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.dice_loss = DiceLoss(softmax=True)

        self.best_mean_dice = 0.0

        self.ds_weights = [0.5, 0.25, 0.125, 0.0625]
        self.ds_weights = [w / sum(self.ds_weights) for w in self.ds_weights]

        self.log_every_n_steps = 25

    def training_step(self, batch):
        image, label = self.get_input(batch)
        assert label.ndim == 5 and label.shape[1] == number_targets, \
            f"Expected one-hot label (B,{number_targets},...), got {tuple(label.shape)}"

        stageA = (self.epoch < curriculum_pretrain_epochs)

        if getattr(self, "_printed_epoch", None) != self.epoch:
            self._printed_epoch = self.epoch
            stage_name = "Stage A (binary)" if stageA else "Stage B (4 regions)"
            fg_frac = float(label[:, 1:].sum(dim=1).clamp(0, 1).mean().item())
            print(f"[Epoch {self.epoch}] {stage_name} | label_shape={tuple(label.shape)} | aorta_union_frac={fg_frac:.4f}")

        x_start = (label * 2) - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        multi_scale_preds = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        total_loss = 0.0

        for i, pred_xstart in enumerate(multi_scale_preds):
            pred_01 = (pred_xstart + 1) / 2

            if i > 0:
                target_5 = torch.nn.functional.interpolate(label, size=pred_01.shape[2:], mode="nearest")
            else:
                target_5 = label

            if stageA:
                target = to_binary_onehot(target_5)
                pred = pred5_to_pred2(pred_01)
            else:
                target = target_5
                pred = pred_01

            loss_dice = self.dice_loss(pred, target)
            target_indices = torch.argmax(target, dim=1).long()
            loss_ce = self.ce(pred, target_indices)
            loss_mse = self.mse(pred, target)

            if not stageA:
                b_epoch = self.epoch - curriculum_pretrain_epochs
                w_bin = min(0.1, 0.1 * (b_epoch / 50.0))
            else:
                w_bin = 0.0

            if stageA:
                scale_loss = loss_dice + loss_ce
            else:
                pred_bin = pred5_to_pred2(pred_01)
                tgt_bin = to_binary_onehot(target_5)
                loss_bin = self.dice_loss(pred_bin, tgt_bin)
                scale_loss = loss_dice + loss_ce + loss_mse + 0.1 * loss_bin

            total_loss += self.ds_weights[i] * scale_loss

        if (self.global_step % self.log_every_n_steps) == 0:
            with torch.no_grad():
                pred0 = (multi_scale_preds[0] + 1) / 2

                pred_bin = pred5_to_pred2(pred0)
                tgt_bin = to_binary_onehot(label)
                dice_bin = 1.0 - self.dice_loss(pred_bin, tgt_bin).item()

                log_payload = {
                    "Training/dice_binary_aorta": float(dice_bin),
                    "global_step": self.global_step
                }

                if not stageA:
                    pred_cls = torch.argmax(pred0, dim=1)
                    tgt_cls = torch.argmax(label, dim=1)

                    dices = []
                    for c in range(1, 5):
                        p = (pred_cls == c).float()
                        t = (tgt_cls == c).float()
                        inter = (p * t).sum()
                        denom = p.sum() + t.sum() + 1e-8
                        dices.append((2.0 * inter / denom))

                    mean_4 = torch.stack(dices).mean().item()
                    log_payload["Training/dice_mean_4_batch"] = float(mean_4)

                wandb.log(log_payload)

        with torch.no_grad():
            if stageA:
                pred0_bin = pred5_to_pred2((multi_scale_preds[0] + 1) / 2)
                tgt0_bin = to_binary_onehot(label)
                train_dice_score = 1.0 - self.dice_loss(pred0_bin, tgt0_bin).item()
            else:
                train_dice_score = 1.0 - self.dice_loss((multi_scale_preds[0] + 1) / 2, label).item()

        self.log("Training/total_loss", total_loss, step=self.global_step)

        wandb.log({
            "Training/total_loss": float(total_loss.item()),
            "Training/dice_score_overall": float(train_dice_score),
            "Training/stageA_binary": int(stageA),
            "global_step": self.global_step
        })

        return total_loss
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"].float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        stageA = (self.epoch < curriculum_pretrain_epochs)

        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        if stageA:
            out_cls = torch.argmax(output, dim=1)
            tgt_cls = torch.argmax(label, dim=1)
            out_bin = (out_cls > 0).cpu().numpy()
            tgt_bin = (tgt_cls > 0).cpu().numpy()
            d = dice(out_bin, tgt_bin)
            return [d, d, d, d]

        output_final = torch.argmax(output, dim=1)
        target_final = torch.argmax(label, dim=1)

        output_np = output_final.cpu().numpy()
        target_np = target_final.cpu().numpy()

        results = []
        for i in range(1, 5):
            results.append(dice(output_np == i, target_np == i))
        return results

    def validation_end(self, mean_val_outputs):
        stageA = (self.epoch < curriculum_pretrain_epochs)

        d1, d2, d3, d4 = mean_val_outputs
        mean_dice = sum(mean_val_outputs) / len(mean_val_outputs)

        if stageA:
            wandb.log({
                "Validation/binary_aorta_dice": float(mean_dice),
                "Validation/epoch": int(self.epoch),
                "global_step": int(self.global_step)
            })
            print(f"[Stage A] Binary Aorta Dice: {mean_dice:.4f}")
            return

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
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )

        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        print(f"Mean Dice Score: {mean_dice:.4f} (Abdom: {d1:.4f}, Arch: {d2:.4f}, Asc: {d3:.4f}, Desc: {d4:.4f})")

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_loader_vascular(data_dir=data_dir, batch_size=batch_size, fold=0, num_workers=4)
    
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
