from torch.utils.data import DataLoader, random_split
from dataloader_dct import PoseKeypointDCTDataset
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load dataset and split into train, validation, and test sets
def load_depth_keypoint_data(depth_dir, keypoint_dir, batch_size=1, train_rate=0.8, validation_rate=0.1):
    dataset = PoseKeypointDCTDataset(depth_dir, keypoint_dir)
    train_size = int(train_rate * len(dataset))
    valid_size = int(validation_rate * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader