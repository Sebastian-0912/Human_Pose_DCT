import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from frequencyDiff import fft_block
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseKeypointDataset(Dataset):

    def __init__(self, depth_dir, keypoint_dir):
        self.depth_dir = depth_dir
        self.keypoint_dir = keypoint_dir
        self.file_pairs = self._get_file_pairs()

    def _get_file_pairs(self):
        depth_files = sorted(
            [f for f in os.listdir(self.depth_dir) if f.endswith('.pkl')])
        keypoint_files = sorted(
            [f for f in os.listdir(self.keypoint_dir) if f.endswith('.pkl')])
        return list(zip(depth_files, keypoint_files))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        depth_file, keypoint_file = self.file_pairs[idx]

        depth_path = os.path.join(self.depth_dir, depth_file)
        keypoint_path = os.path.join(self.keypoint_dir, keypoint_file)

        with open(depth_path, 'rb') as f:
            depth_sequence = pickle.load(f)  # (frames, 3, 64, 64)

        with open(keypoint_path, 'rb') as f:
            # (frames, 17, 3)( 3 is x, y, confidence)
            keypoint_sequence = pickle.load(f)

        # Discard the third element in the last dimension
        # (frames, 17, 2)( 2 is x, y)
        keypoint_sequence = keypoint_sequence[:, :, :2]

        fre_sequence = fft_block(depth_sequence)  # (frames, 1, 64, 64)

        # # Normalize depth maps
        depth_sequence = torch.tensor(depth_sequence,
                                      dtype=torch.float32) / 255.0
        # Normalize depth maps
        fre_sequence = torch.tensor(fre_sequence, dtype=torch.float32) / 255.0

        keypoint_sequence = torch.tensor(keypoint_sequence,
                                         dtype=torch.float32)

        return depth_sequence, fre_sequence, keypoint_sequence


class PoseKeypointCNN(nn.Module):

    def __init__(self, num_keypoints=17):
        super(PoseKeypointCNN, self).__init__()

        self.depth_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Sequential(nn.Linear(64 * 2 * 16 * 16, 512), nn.ReLU(),
                                nn.Linear(512, num_keypoints * 2))

    def forward(self, depth_frame, freq_frame):
        depth_features = self.depth_conv(depth_frame)
        freq_features = self.freq_conv(freq_frame)
        combined_features = torch.cat((depth_features, freq_features), dim=0)
        combined_features = combined_features.view(-1, 64 * 2 * 16 * 16)
        output = self.fc(combined_features)
        return output


if __name__ == '__main__':

    # Paths to the directories containing the depth maps and key points
    depth_dir = "./dataset/SUSTech1K-Reorganized/depth_map"
    keypoint_dir = "./dataset/SUSTech1K-Reorganized/key_points"

    # Create dataset and dataloader
    PoseDataset = PoseKeypointDataset(depth_dir, keypoint_dir)

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(PoseDataset))
    valid_size = int(0.15 * len(PoseDataset))
    test_size = len(PoseDataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(
        PoseDataset, [train_size, valid_size, test_size])

    # Create dataloaders
    batch_size = 1
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # # get some random training images
    # dataiter = iter(train_loader)
    # depth, frequence, keypoint = next(dataiter)
    # print(depth.shape)
    # print(frequence.shape)
    # print(keypoint.shape)

    # Initialize model, loss function, and optimizer
    model = PoseKeypointCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for depth_sequences, freq_sequences, keypoints_sequences in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            batch_loss = 0.0
            for i in range(batch_size):
                depth_sequence = depth_sequences[i]
                freq_sequence = freq_sequences[i]
                keypoints_sequence = keypoints_sequences[i]

                sequence_length = len(depth_sequence)
                for depth_frame, freq_frame, keypoints in zip(
                        depth_sequence, freq_sequence, keypoints_sequence):
                    depth_frame, freq_frame, keypoints = depth_frame.to(
                        device), freq_frame.to(device), keypoints.to(device)
                    output = model(depth_frame, freq_frame)
                    loss = criterion(output, keypoints.view(1, -1))
                    batch_loss += loss

            batch_loss /= batch_size
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()

        train_losses.append(running_loss / len(train_loader))

        if (epoch + 1) % 5 == 0:
            # Save the model
            torch.save(model.state_dict(),
                       f'./model_pth/poseCnn_pth/epoch_{epoch}_cnn.pth')

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for depth_sequences, freq_sequences, keypoints_sequences in tqdm(
                    valid_loader,
                    desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                batch_loss = 0.0
                for i in range(batch_size):
                    depth_sequence = depth_sequences[i]
                    freq_sequence = freq_sequences[i]
                    keypoints_sequence = keypoints_sequences[i]

                    sequence_length = len(depth_sequence)
                    for depth_frame, freq_frame, keypoints in zip(
                            depth_sequence, freq_sequence, keypoints_sequence):
                        depth_frame, freq_frame, keypoints = depth_frame.to(
                            device), freq_frame.to(device), keypoints.to(
                                device)
                        output = model(depth_frame, freq_frame)
                        loss = criterion(output, keypoints.view(1, -1))
                        batch_loss += loss

                batch_loss /= batch_size
                valid_loss += batch_loss.item()

        valid_losses.append(valid_loss / len(valid_loader))

        print(
            f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}'
        )

    print('Training Finished')

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
    plt.savefig("./figure/loss_curve.png")

    # Save the model
    torch.save(model.state_dict(),
               './model_pth/poseCnn_pth/pose_keypoint_cnn.pth')

    # Testing loop
    model.eval()
    test_loss = 0.0
    test_loss_array = []
    with torch.no_grad():
        for depth_sequences, freq_sequences, keypoints_sequences in tqdm(
                test_loader, desc="Testing"):
            batch_loss = 0.0
            for i in range(batch_size):
                depth_sequence = depth_sequences[i]
                freq_sequence = freq_sequences[i]
                keypoints_sequence = keypoints_sequences[i]

                sequence_length = len(depth_sequence)
                for depth_frame, freq_frame, keypoints in zip(
                        depth_sequence, freq_sequence, keypoints_sequence):
                    depth_frame, freq_frame, keypoints = depth_frame.to(
                        device), freq_frame.to(device), keypoints.to(device)
                    output = model(depth_frame, freq_frame)
                    loss = criterion(output, keypoints.view(1, -1))
                    batch_loss += loss
            test_loss_array.append(batch_loss.item())
            batch_loss /= batch_size
            test_loss += batch_loss.item()

    print(f'Test Loss: {test_loss / len(test_loader)}')

    # Save the test loss in a pickle file
    with open('test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss_array, f)
