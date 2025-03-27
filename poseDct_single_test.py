import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader_dct import PoseKeypointDCTDataset,idct_block
from model.poseDct import PoseKeypointDCT
# from backup_model.poseDct_v3 import PoseKeypointDCT
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to the directories containing the depth maps and key points
# depth_dir = "./dataset/SUSTech1K-Reorganized/depth_map"
# keypoint_dir = "./dataset/SUSTech1K-Reorganized/key_points"
depth_dir = "./dataset/SUSTech1K-Reorganized-divided/depth_map_nm"
keypoint_dir = "./dataset/SUSTech1K-Reorganized-divided/key_points_nm"

# Load the dataset
PoseDataset = PoseKeypointDCTDataset(depth_dir, keypoint_dir)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(PoseDataset))
valid_size = int(0.1 * len(PoseDataset))
test_size = len(PoseDataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    PoseDataset, [train_size, valid_size, test_size])

test_loader = DataLoader(test_dataset , batch_size=1, shuffle=False)

# Initialize and train the model
model = PoseKeypointDCT()
model = model.to(device)
# model_path = './model_pth/poseDct_Cnn64_pth/epoch_12_dct.pth'
# model_path ="/home/sebastian/Desktop/HumanPose_CNN/model_pth/poseDct_v3_pth/epoch_4_dct.pth"
model_path ="/home/sebastian/Desktop/HumanPose_CNN/model_pth/poseDct_v5_pth/epoch_9_dct.pth"

model.load_state_dict(torch.load(model_path))
model.eval()
count = 0
# Validation and extraction of the first frame
with torch.no_grad():
    for freq_sequence, keypoint_sequence in tqdm(test_loader, desc="Testing"):
        if(count!=34):
            count+=1
            continue
        freq_sequence = freq_sequence.to(device)
        keypoint_sequence = keypoint_sequence.to(device)
        # print(freq_sequence.shape)
        # print(keypoint_sequence.shape)
        
        frame_num = keypoint_sequence.shape[1]
        keypoint_sequence = keypoint_sequence.view(-1)  # (frames, 17, 2)
        
        # Forward pass
        outputs = model(freq_sequence)
        outputs = idct_block(outputs, target_length=frame_num * 2).reshape(frame_num, 17, 2)
        keypoint_sequence = keypoint_sequence.view(frame_num, -1, 2)  # (frames, 17, 2)
        print(keypoint_sequence.shape)
        # print(outputs.shape)
        # Extract the first frame
        keypoints_pred = outputs[1].cpu().numpy()  # Predicted keypoints for the first frame
        keypoint_sequence = keypoint_sequence[1].cpu().numpy()  # Convert to numpy for visualization
        # keypoints_pred = outputs.cpu().numpy()  # Predicted keypoints for the first frame
        # keypoint_sequence = keypoint_sequence.cpu().numpy()  # Convert to numpy for visualization
        print(keypoints_pred.shape)
        print(keypoint_sequence.shape)
        break
        

keypoint_path = "./cross_model_demo/keypoint.pkl"  # Ground truth keypoint file path
dct_predicted_path = "./cross_model_demo/dct.pkl"  # DCT model predicted keypoint file path

with open(keypoint_path, 'wb') as f:
    pickle.dump(keypoint_sequence,f)  # (17, 3)

with open(dct_predicted_path, 'wb') as f:
    pickle.dump(keypoints_pred,f)  # (17, 3)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

# Invert the y-axis coordinates for both prediction and ground truth
keypoints_pred[:, 1] = -keypoints_pred[:, 1]
keypoint_sequence[:, 1] = -keypoint_sequence[:, 1]

# Define the COCO skeleton connections with 1-based indexing (as per COCO format)
coco_skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# Convert COCO format to zero-indexed for Python
connections = [[start - 1, end - 1] for start, end in coco_skeleton]

# Plot the skeleton by connecting the joints for ground truth
for start, end in connections:
    ax.plot([
        keypoint_sequence[start, 0],
        keypoint_sequence[end, 0]
    ], [
        keypoint_sequence[start, 1],
        keypoint_sequence[end, 1]
    ],
            'b-',
            label='Ground Truth' if start == 0 and end == 1 else "")

# Plot the skeleton by connecting the joints for prediction
for start, end in connections:
    ax.plot([keypoints_pred[start, 0], keypoints_pred[end, 0]],
            [keypoints_pred[start, 1], keypoints_pred[end, 1]],
            'r-',
            label='Predicted' if start == 0 and end == 1 else "")

# Plot each joint
ax.scatter(keypoint_sequence[:, 0],
           keypoint_sequence[:, 1],
           c='b',
           marker='o',
           label='Ground Truth Joints')
ax.scatter(keypoints_pred[:, 0],
           keypoints_pred[:, 1],
           c='r',
           marker='x',
           label='Predicted Joints')

# Set axis labels, aspect ratio, and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

# Save the plot and show it
plt.savefig('./figure/demo_dct.png')
# plt.show()