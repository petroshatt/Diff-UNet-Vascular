# nohup env CUDA_VISIBLE_DEVICES=2 python -u test.py > test_log.txt 2>&1 &

import numpy as np
import torch 
import torch.nn as nn 
import os
import nibabel as nib
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall

from dataset.vascular_data_utils_multi_label import get_test_loader_vascular
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

set_determinism(123)

data_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/imagesTs"
labels_data_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/labelsTs"
logdir = "/data/pchatzi/Diff-UNet/Vascular/logs_vascular/diffusion_seg/model/best_model_0.7675.pt" 
output_save_dir = "/data/pchatzi/Diff-UNet/Vascular/logs_vascular/diffusion_seg/predictions/best_model_0.7675_predictions"

os.makedirs(output_save_dir, exist_ok=True)

batch_size = 1
device = "cuda:0"
number_modality = 2
number_targets = 5

def compute_uncer(pred_out):
    pred_out = torch.softmax(pred_out, dim=1)
    uncer_out = - pred_out * torch.log(pred_out + 1e-6)
    return uncer_out.sum(dim=1)

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas, model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas, model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "ddim_sample":
            current_bs = image.shape[0]
            embeddings = self.embed_model(image)
            uncer_step = 4
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(
                    self.model, (current_bs, number_targets, 96, 96, 96), 
                    model_kwargs={"image": image, "embeddings": embeddings})
                )

            sample_return = torch.zeros((current_bs, number_targets, 96, 96, 96)).to(image.device)
            
            for index in range(10):
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step

                uncer = compute_uncer(uncer_out).to(image.device)
                uncer = uncer.unsqueeze(1)

                time_val = torch.tensor((index + 1) / 10, device=image.device)
                w = torch.exp(torch.sigmoid(time_val) * (1 - uncer))
                
                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].to(image.device)

            sample_return = (sample_return + 1) / 2
            return sample_return

class VascularTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", logdir="./logs/"):
        super().__init__(env_type, max_epochs, batch_size, device=device, logdir=logdir)
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96], 
            sw_batch_size=4, 
            overlap=0.5,
            device=torch.device(device) 
        )
        self.model = DiffUNet()

    def save_and_evaluate(self, batch, output_tensor, save_path, case_index):
        output_indices = torch.argmax(output_tensor, dim=1).cpu().numpy()[0]
        label_np = batch["label"].cpu().numpy()[0]

        case_metrics = []
        for i in range(1, number_targets):
            o = (output_indices == i).astype(np.float32)

            if label_np.ndim == 4 and label_np.shape[0] == number_targets:
                t = label_np[i].astype(np.float32)
            elif label_np.ndim == 4 and label_np.shape[0] == 1:
                t = (label_np[0] == i).astype(np.float32)
            else:
                t = (label_np == i).astype(np.float32)

            t = np.squeeze(t)
            o = np.squeeze(o)

            if o.shape != t.shape:
                print(f"ERROR: Shape mismatch Case {i} | Pred: {o.shape} | Label: {t.shape}")
                raise ValueError(f"Check your volume dimensions for class {i}")
            
            case_metrics.append([
                dice(o, t), 
                hausdorff_distance_95(o, t), 
                recall(o, t)
            ])

        image_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/imagesTs"
        labels_dir = "/data/pchatzi/Diff-UNet/Vascular/dataset/Dataset001_Vascular/labelsTs"
        
        image_files = sorted([f for f in os.listdir(image_dir) if "_0000.nii.gz" in f])
        
        if case_index < len(image_files):
            case_id = image_files[case_index].split("_0000")[0].split(".nii.gz")[0]
        else:
            case_id = f"unknown_index_{case_index}"

        label_file_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        if os.path.exists(label_file_path):
            ref_img = nib.load(label_file_path)
            affine = ref_img.affine
            header = ref_img.header
        else:
            print(f"Warning: {label_file_path} not found. Using identity.")
            affine = np.eye(4)
            header = None

        pred_to_save = output_indices.transpose(2, 1, 0).astype(np.uint8)

        save_name = os.path.join(save_path, f"{case_id}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred_to_save, affine, header), save_name)
        
        return case_metrics, case_id

if __name__ == "__main__":
    print(f"Searching for test images in: {data_dir}")
    test_ds = get_test_loader_vascular(data_dir=data_dir, batch_size=batch_size)
    
    trainer = VascularTrainer(env_type="pytorch", max_epochs=1, batch_size=batch_size, device=device)
    
    print(f"Loading weights from {logdir}...")
    trainer.load_state_dict(logdir)
    trainer.model.to(device)
    trainer.model.eval()

    classes = ["Abdominal", "Aortic Arch", "Ascending", "Descending"]
    all_metrics = []

    print(f"Found {len(test_ds)} cases in test set.")
    print(f"Starting Inference & Evaluation...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_ds):
            image = batch["image"].unsqueeze(0).to(device)
            
            output = trainer.window_infer(image, trainer.model, pred_type="ddim_sample")
            
            case_metrics, case_id = trainer.save_and_evaluate(batch, output, output_save_dir, i)
            all_metrics.append(case_metrics)

            print(f"Inference Case: {case_id} ({i+1}/{len(test_ds)})")
            
            for idx, c_name in enumerate(classes):
                d, hd, r = case_metrics[idx]
                print(f"  {c_name:12} | Dice: {d:.4f} | HD95: {hd:.2f} | Recall: {r:.4f}")

    if len(all_metrics) > 0:
        all_metrics = np.array(all_metrics) 
        mean_metrics = np.mean(all_metrics, axis=0)

        print("\n" + "="*60)
        print(f"{'CLASS':<15} | {'DICE':<8} | {'HD95':<8} | {'RECALL':<8}")
        print("-" * 60)
        for i, c_name in enumerate(classes):
            print(f"{c_name:<15} | {mean_metrics[i,0]:.4f}   | {mean_metrics[i,1]:.2f}   | {mean_metrics[i,2]:.4f}")
        print("="*60)
