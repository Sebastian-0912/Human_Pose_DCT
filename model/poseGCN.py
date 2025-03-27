import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import itertools

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_complete_graph(num_nodes):
    edges = list(itertools.combinations(range(num_nodes), 2))
    edges += [(j, i) for i, j in edges]  # Add both directions to make it undirected
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return edge_index

class PoseKeypointGAT(nn.Module):
    def __init__(self, num_keypoints=17, output_length=50, in_features=50, hidden_features=128, heads=4):
        super(PoseKeypointGAT, self).__init__()
        self.num_keypoints = num_keypoints
        self.output_length = output_length
        self.edge_index = create_complete_graph(num_keypoints)
        # First GAT layer
        self.gat1 = GATConv(in_channels=in_features, out_channels=hidden_features, heads=heads, concat=True)
        # Second GAT layer with output set to DCT representation dimension
        self.gat2 = GATConv(in_channels=hidden_features * heads, out_channels=output_length, heads=1, concat=False)
        self.fc1 = nn.Linear(self.num_keypoints*output_length, self.num_keypoints*output_length)  # Fully connected layer for additional transformation

    def forward(self, x):
        # Apply first GAT layer with ReLU activation
        x = self.gat1(x, self.edge_index)
        x = F.relu(x)
        
        # Apply second GAT layer to get the final DCT representation output
        x = self.gat2(x, self.edge_index)
        x = F.relu(x)
        x = x.view(-1,self.num_keypoints*self.output_length)
        x = self.fc1(x)
        
        # Reshape to ensure output is (num_keypoints, output_length) per batch
        x = x.view(-1, self.num_keypoints, self.output_length)
        return x


class PoseKeypointGAT_residual(nn.Module):
    def __init__(self, num_keypoints=17, output_length=50, in_features=50, hidden_features=128, out_features=50, heads=4):
        super(PoseKeypointGAT_residual, self).__init__()
        self.num_keypoints = num_keypoints
        self.output_length = output_length
        self.edge_index = create_complete_graph(num_keypoints)
        
        self.gat1 = GATConv(in_channels=in_features, out_channels=hidden_features, heads=heads, concat=True)
        self.gat2 = GATConv(in_channels=hidden_features * heads, out_channels=hidden_features, heads=heads, concat=True)
        self.gat3 = GATConv(in_channels=hidden_features * heads, out_channels=out_features, heads=1, concat=False)  # final layer, output heads=1

        self.fc1 = nn.Linear(self.num_keypoints*output_length, self.num_keypoints*output_length)  # Fully connected layer for additional transformation
        self.norm1 = nn.LayerNorm(hidden_features * heads)
        self.norm2 = nn.LayerNorm(512)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Initial GAT layer with activation and normalization
        x = F.relu(self.gat1(x, self.edge_index))
        x = self.norm1(x)
        x = self.dropout(x)
        
        # Second GAT layer with residual connection
        residual = x
        x = F.relu(self.gat2(x, self.edge_index))
        x = self.norm2(x + residual)
        x = self.dropout(x)

        # Final GAT layer for output, no activation for smoother output
        x = self.gat3(x, self.edge_index)
        
        # Fully connected layer to refine output
        x = x.view(-1,self.num_keypoints*self.output_length)
        x = self.fc1(x)
        x = x.view(-1, self.num_keypoints, self.output_length)
        return x