# import torch
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# def calculate_mpjpe(preds, targets, num_keypoints=17):
#     preds = preds.view(-1, num_keypoints, 2)  # (frames, 17, 2)
#     targets = targets.view(-1, num_keypoints, 2)
#     return torch.mean(torch.norm(preds - targets, dim=2)).item()
  
# c = torch.tensor([[[ 1, 0], [ 3,0]]] , dtype=torch.float)  
  
# print(torch.mean(torch.norm(c,dim=2)).item())

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch.optim import Adam

# # Define a GAT model with attention heads
# class FullGATExample(torch.nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, heads=2):
#         super(FullGATExample, self).__init__()
#         self.gat1 = GATConv(in_features, hidden_features, heads=heads)
#         self.gat2 = GATConv(hidden_features * heads, out_features, heads=1)

#     def forward(self, x, edge_index):
#         x = self.gat1(x, edge_index)
#         x = torch.relu(x)
#         x = self.gat2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# # Define node features and edge connections (edge_index)
# node_features = torch.tensor([[1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.float32)
# edge_index = torch.tensor([
#     [0, 1, 1, 2, 2, 3, 3, 0],
#     [1, 0, 2, 1, 3, 2, 0, 3]
# ], dtype=torch.long)
# labels = torch.tensor([0, 1, 0, 1])  # Ground truth labels for each node

# # Initialize model, optimizer, and loss function
# model = FullGATExample(in_features=3, hidden_features=4, out_features=2, heads=2)
# optimizer = Adam(model.parameters(), lr=0.01)
# loss_fn = F.nll_loss  # Negative log-likelihood loss for classification

# # Training loop
# for epoch in range(50):
#     optimizer.zero_grad()
#     out = model(node_features, edge_index)
#     loss = loss_fn(out, labels)
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")
import torch
import torch.nn as nn

# # Define the linear layer
# linear = nn.Linear(50, 50)

# # Define the input tensor
# x = torch.randn(17, 50)

# # Pass x through the linear layer
# output = linear(x)

# print(output.shape)  # Output: torch.Size([17, 50])

x = torch.tensor([[[1, 10, 7],
                  [2, 5, 9],
                  [2, 5, 19]],
                  [[1, 10, 7],
                  [2, 5, 9],
                  [2, 5, 19]]])
print(x.size())
max_values, indices = torch.max(x, dim=1)  # Find max along dim=1 (columns)
print(max_values.size())  # Output: tensor([7, 9])
print(indices)  # Output: tensor([2, 2])  # Index of max in each row
