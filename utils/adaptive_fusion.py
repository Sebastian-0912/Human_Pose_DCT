import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AdaptiveFusion, self).__init__()
        self.weight_layer = nn.Linear(feature_dim * 2, 2)  # Map concatenated features to 2 scalars

    def forward(self, F_PLF, F_DF):
        # Concatenate along the feature dimension
        F_concat = torch.cat([F_PLF, F_DF], dim=-1)  # Shape: (B, ..., 2 * feature_dim)
        
        # Compute adaptive weights
        alpha = torch.softmax(self.weight_layer(F_concat), dim=-1)  # Shape: (B, ..., 2)

        # Split the weights
        alpha_PLF, alpha_DF = alpha[..., 0:1], alpha[..., 1:2]  # Keep dimensions for broadcasting

        # Fuse features
        F_fused = alpha_PLF * F_PLF + alpha_DF * F_DF
        return F_fused
