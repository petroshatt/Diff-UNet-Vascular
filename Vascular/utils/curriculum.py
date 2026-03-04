import torch

def to_binary_onehot(label_5ch: torch.Tensor) -> torch.Tensor:
    """
    (B,5,...) one-hot -> (B,2,...) one-hot where fg is union of classes 1..4
    """
    fg = torch.clamp(label_5ch[:, 1:].sum(dim=1, keepdim=True), 0, 1)
    bg = 1.0 - fg
    return torch.cat([bg, fg], dim=1)

def pred5_to_pred2(pred_5ch: torch.Tensor) -> torch.Tensor:
    """
    (B,5,...) probabilities -> (B,2,...) where fg is sum of classes 1..4
    """
    bg = pred_5ch[:, 0:1]
    fg = pred_5ch[:, 1:].sum(dim=1, keepdim=True)
    return torch.cat([bg, fg], dim=1)