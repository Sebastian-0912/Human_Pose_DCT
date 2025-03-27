import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader_dct import PoseKeypointDCTDataset,idct_block
from backup_model.poseDct_v3 import PoseKeypointDCT
from utils.mertics import calculate_mae,calculate_mpjpe,calculate_rmse
# from model.poseDct import PoseKeypointDCT

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths to the directories containing the depth maps and key points
# depth_dir = "./dataset/SUSTech1K-Reorganized/depth_map"
# keypoint_dir = "./dataset/SUSTech1K-Reorganized/key_points"
depth_dir = "./dataset/SUSTech1K-Reorganized-divided/depth_map_nm"
keypoint_dir = "./dataset/SUSTech1K-Reorganized-divided/key_points_nm"

# Load the dataset
PoseDataset = PoseKeypointDCTDataset(depth_dir, keypoint_dir)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(PoseDataset))
valid_size = int(0.15 * len(PoseDataset))
test_size = len(PoseDataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    PoseDataset, [train_size, valid_size, test_size])

test_loader = DataLoader(test_dataset , batch_size=1, shuffle=False)

# Initialize and train the model
model = PoseKeypointDCT()
model = model.to(device)
# model_path = './model_pth/poseDct_v2_pth/epoch_16_dct.pth'
# model_path ="/home/sebastian/Desktop/HumanPose_CNN/model_pth/poseDct_v4_pth/epoch_14_dct.pth"
# model_path ="/home/sebastian/Desktop/HumanPose_CNN/model_pth/poseDct_v3_pth/epoch_4_dct.pth"
model_path ="/home/sebastian/Desktop/HumanPose_CNN/model_pth/poseDct_v5_pth/epoch_9_dct.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize variables to store test loss and metrics
mae_values = []
rmse_values = []
mpjpe_values = []

# Validation
val_loss = 0.0
with torch.no_grad():
    for freq_sequence, keypoint_sequence in tqdm(test_loader ,desc="Testing"):
        freq_sequence = freq_sequence.to(device)
        
        frame_num =keypoint_sequence.shape[1]
        
        keypoint_sequence = keypoint_sequence.to(device).view(-1)

        # Forward pass
        outputs = model(freq_sequence)
        outputs = idct_block(outputs,target_length=frame_num*2).reshape(-1)
        
        # Calculate metrics
        mae = calculate_mae(outputs, keypoint_sequence)
        rmse = calculate_rmse(outputs, keypoint_sequence)
        mpjpe = calculate_mpjpe(outputs, keypoint_sequence, num_keypoints=17)
        
        # Store the metrics
        mae_values.append(mae)
        rmse_values.append(rmse)
        mpjpe_values.append(mpjpe)

# Calculate average metrics
avg_mae = np.mean(mae_values)
avg_rmse = np.mean(rmse_values)
avg_mpjpe = np.mean(mpjpe_values)

print(f"Average MAE: {avg_mae:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average MPJPE: {avg_mpjpe:.4f}")

# Plotting the metrics
metrics = ['MAE', 'RMSE', 'MPJPE']
values = [avg_mae, avg_rmse, avg_mpjpe]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])
plt.title('Test Metrics')
plt.ylabel('Value')
plt.xlabel('Metric')
plt.ylim(0, max(values) * 1.2)  # Adjust y-axis for better visualization
# plt.savefig('./figure/test_metrics.png')
plt.show()