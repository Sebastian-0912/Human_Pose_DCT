import torch
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PoseKeypointDCT(nn.Module):

    def __init__(self, num_keypoints=17, output_length=50):
        # input width depens on the dct_block output, here is 40
        super(PoseKeypointDCT, self).__init__()
        self.output_length = output_length
        self.number_keypoints = num_keypoints
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        # output shape (32, 1024, 10)
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 1024 * 10, 1024), nn.ReLU(),
            # nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, num_keypoints * self.output_length))
        # Output shape should be (17, output_length) that stands for the frequency feature

    def forward(self, depth_freq):
        freq_features = self.freq_conv(depth_freq)
        batch_size = freq_features.shape[0]
        freq_features = freq_features.view(batch_size,-1)
        # print("freq_features",freq_features.shape)
        keypoint_freq = self.fc(freq_features)
        keypoint_freq = keypoint_freq.view(batch_size, self.number_keypoints, self.output_length)  # Reshape to (number_keypoints,target_length)
        # print(keypoint_freq.shape)

        return  keypoint_freq