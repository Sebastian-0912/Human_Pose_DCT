import torch

# MAE, RMSE, and MPJPE calculation functions
def calculate_mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def calculate_rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def calculate_mpjpe(preds, targets, num_keypoints=17):
    preds = preds.view(-1, num_keypoints, 2)  # (frames, 17, 2)
    targets = targets.view(-1, num_keypoints, 2)
    return torch.mean(torch.norm(preds - targets, dim=2)).item()