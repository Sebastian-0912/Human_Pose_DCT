# it is v5 model now 
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader_dct import PoseKeypointDCTDataset,idct_block
from model.poseDct import PoseKeypointDCT
from custom_loss.mpjpe import MPJPELoss

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

batch_size =1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# Initialize and train the model
model = PoseKeypointDCT()
# model_path = './model_pth/epoch_9_dct.pth'
# model.load_state_dict(torch.load(model_path))

# Set device
model = model.to(device)

# Loss and optimizer
criterion = MPJPELoss() # for v3
# criterion = nn.MSELoss() # for v4
optimizer = optim.Adam(model.parameters(), lr=0.0005)


train_losses = []
valid_losses = []

# Training loop with validation
num_epochs = 12
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for freq_sequence, keypoint_sequence in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        freq_sequence = freq_sequence.to(device)
        # print(freq_sequence.shape)
        frame_num =keypoint_sequence.shape[1]
        
        keypoint_sequence = keypoint_sequence.to(device).view(batch_size,-1)  # Flatten for MSELoss
        # print(freq_sequence.shape)
        
        # Forward pass
        outputs = model(freq_sequence)
        outputs = idct_block(outputs,target_length=frame_num*2).reshape(batch_size,-1)
        loss = criterion(outputs, keypoint_sequence)
        print(loss)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # print(torch.cuda.memory_summary())
        optimizer.step()

        running_loss += loss.item()
        
    train_losses.append(running_loss / len(train_loader))
        
    if (epoch + 1) % 3 == 0:
        # Save the model
        torch.save(model.state_dict(),f'./model_pth/poseDct_v5_pth/epoch_{epoch+1}_dct.pth')
        
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for freq_sequence, keypoint_sequence in tqdm(valid_loader ,desc=f'Epoch {epoch+1}/{num_epochs}'):
            freq_sequence = freq_sequence.to(device)
            
            frame_num =keypoint_sequence.shape[1]
            
            keypoint_sequence = keypoint_sequence.to(device).view(-1)

            # Forward pass
            outputs = model(freq_sequence)
            outputs = idct_block(outputs,target_length=frame_num*2).reshape(-1)
            loss = criterion(outputs, keypoint_sequence)
            val_loss += loss.item()
            
        valid_losses.append(val_loss / len(valid_loader))
    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(valid_loader):.4f}'
    )

# Plot validation curve
plt.plot(range(1, num_epochs + 1),
          train_losses,
          label='Train Loss',
          marker='o')
plt.plot(range(1, num_epochs + 1),
          valid_losses,
          label='Validation Loss',
          marker='v')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./figure/loss_curve_dct_v5.png")  