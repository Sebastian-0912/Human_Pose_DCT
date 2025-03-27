import torch
import os
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_dct import dct, idct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class PoseKeypointDCTDataset(Dataset):

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
        # print(depth_path)
        # print(keypoint_path)
        with open(depth_path, 'rb') as f:
            depth_sequence = pickle.load(f)  # (frames, 3, 64, 64)

        with open(keypoint_path, 'rb') as f:
            keypoint_sequence = pickle.load(f)  # (frames, 17, 3)

        # only input ten frames at most
        if(depth_sequence.shape[0]>20):
            depth_sequence = depth_sequence[:20,...]
            keypoint_sequence= keypoint_sequence[:20,...]
            
        # Discard the third element in the last dimension
        keypoint_sequence = keypoint_sequence[:, :, :2]  # (frames, 17, 2)
        
        # Apply DCT to the depth sequence, and already become tensor
        freq_sequence = dct_block(depth_sequence)  # (3, 4096, 40)

        keypoint_sequence = torch.tensor(keypoint_sequence,
                                         dtype=torch.float32)
        
        return freq_sequence, keypoint_sequence
      
# Function to apply DCT to each 1D array extracted from depth images //ok
def dct_block(depth_sequence, _norm='ortho', target_length =40):
    """
    input: depth_sequence shape: (frames, 3, 64, 64)
    output: transformed_depth shape: (3, 64 * 64, frames)
    """
    frames, channels, H, W = depth_sequence.shape
    depth_sequence = depth_sequence.reshape(frames, channels, H * W)
    transformed_depth = depth_sequence.transpose(1, 2, 0)  # Shape: (channels, H*W, frames)
    transformed_depth = torch.tensor(transformed_depth,dtype=torch.float32)
    transformed_depth = dct(transformed_depth, norm=_norm)
    
    # Adjust the last dimension to the target length
    current_length = transformed_depth.shape[-1]
    
    if current_length < target_length:
        # Pad with zeros if the current length is less than the target length
        padding = torch.zeros(transformed_depth.shape[:-1] + (target_length - current_length,))
        transformed_depth = torch.cat((transformed_depth, padding), dim=-1)
    elif current_length > target_length:
        # Truncate if the current length is greater than the target length
        transformed_depth = transformed_depth[..., :target_length]
    
    return transformed_depth


# Function to apply IDCT to the predicted keypoint frequency
def idct_block(keypoint_freq, _norm='ortho', target_length =60):
    """
    input: keypoint_frequency shape: (17, 2*frames)
    output: keypoint_sequence shape: (17, target_length )
    """
    
    # Adjust the last dimension to the target length
    current_length = keypoint_freq.shape[-1]
    
    if current_length < target_length:
        # Pad with zeros if the current length is less than the target length
        padding = torch.zeros(keypoint_freq.shape[:-1] + (target_length - current_length,)).to(device)
        keypoint_freq = torch.cat((keypoint_freq, padding), dim=-1)
        
    # idct() first to remain more feature
    keypoint_sequence = idct(keypoint_freq, norm=_norm)
    
    if current_length > target_length:
        # Truncate if the current length is greater than the target length
        keypoint_sequence = keypoint_sequence[:,:, :target_length]
    # print(keypoint_sequence.shape)
    return keypoint_sequence
