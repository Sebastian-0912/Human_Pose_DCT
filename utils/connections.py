# This file define the graph we use in the COCO format.
# 0: Nose
# 1: Left Eye
# 2: Right Eye
# 3: Left Ear
# 4: Right Ear
# 5: Left Shoulder
# 6: Right Shoulder
# 7: Left Elbow
# 8: Right Elbow
# 9: Left Wrist
# 10: Right Wrist
# 11: Left Hip
# 12: Right Hip
# 13: Left Knee
# 14: Right Knee
# 15: Left Ankle
# 16: Right Ankle

import torch

# Limb graph in undirect format
UNDIRECT_LIMB_GRAPH = [
  # left arm and right leg complete_graph
    (5, 7), (5, 9), (5, 12), (5, 14), (5, 16),
    (7, 9), (7, 12), (7, 14), (7, 16),
    (9, 12), (9, 14), (9, 16),
    (12, 14), (12, 16),
    (14, 16),
    
  # right arm and left leg complete_graph
    (6, 8), (6, 10), (6, 11), (6, 13), (6, 15),
    (8, 10), (8, 11), (8, 13), (8, 15),
    (10, 11), (10, 13), (10, 15),
    (11, 13), (11, 15),
    (13, 15)
]

# Human graph in undirect format
UNDIRECT_HUMAN_GRAPH = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5),  # Face
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Limb graph in PyG format
DIRECT_LIMB_GRAPH = [
  [ 
    5,  7,  5,  9,  5, 12,  5, 14,  5, 16,  7,  9,  7, 12,  7, 14,  7, 16,
    9, 12,  9, 14,  9, 16, 12, 14, 12, 16, 14, 16,  6,  8,  6, 10,  6, 11,
    6, 13,  6, 15,  8, 10,  8, 11,  8, 13,  8, 15, 10, 11, 10, 13, 10, 15,
    11, 13, 11, 15, 13, 15
  ], # source node
  [ 
    7,  5,  9,  5, 12,  5, 14,  5, 16,  5,  9,  7, 12,  7, 14,  7, 16,  7,
    12,  9, 14,  9, 16,  9, 14, 12, 16, 12, 16, 14,  8,  6, 10,  6, 11,  6,
    13,  6, 15,  6, 10,  8, 11,  8, 13,  8, 15,  8, 11, 10, 13, 10, 15, 10,
    13, 11, 15, 11, 15, 13
  ] # target node
]

# Human graph in PyG format
DIRECT_HUMAN_GRAPH = [
  [ 
    0,  1,  0,  2,  1,  2,  1,  3,  2,  4,  4,  6,  3,  5,  5,  7,  7,  9,
    6,  8,  8, 10,  5,  6,  5, 11,  6, 12, 11, 12, 11, 13, 13, 15, 12, 14,
    14, 16
  ], # source node
  [ 
    1,  0,  2,  0,  2,  1,  3,  1,  4,  2,  6,  4,  5,  3,  7,  5,  9,  7,
    8,  6, 10,  8,  6,  5, 11,  5, 12,  6, 12, 11, 13, 11, 15, 13, 14, 12,
    16, 14
  ]  # target node
]


# return direct graph in PyG format
def knn_graph(joints_feature: torch.Tensor, K: int):
    """
    Constructs a KNN-based dynamic graph for joints.

    Args:
        joints_feature (torch.Tensor): Shape (J, F), where J is the number of joints, F is feature dimension.
        K (int): Number of nearest neighbors to connect each joint.

    Returns:
        edge_index (torch.Tensor): Shape (2, J*K), representing directed edges.
    """
    J = joints_feature.shape[0]  # Number of joints

    # Compute pairwise Euclidean distance matrix (J, J)
    dist_matrix = torch.cdist(joints_feature, joints_feature, p=2)  # L2 distance

    # Get the top-K nearest neighbors (excluding self-loop)
    knn_indices = dist_matrix.argsort(dim=1)[:, 1:K+1]  # (J, K), ignore self (first column is self)

    # Create edge index (source -> target)
    source_nodes = torch.arange(J).repeat(K)  # Repeat each node J times
    target_nodes = knn_indices.flatten()  # Flatten K-neighbors for each node

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)  # Shape (2, J*K)

    return edge_index

# # Example usage
# joints_feature = torch.rand(17, 3)  # 17 joints, each with a 3D feature (e.g., x, y, z)
# K = 3  # Connect each joint to 3 nearest neighbors
# edge_index = knn_graph(joints_feature, K)

# print("Edge Index:\n", edge_index)

def convert_to_pyg_format(undirected_graph):
    """Converts an undirected graph to PyG edge_index format."""
    edge_list = []
    for u, v in undirected_graph:
        edge_list.append((u, v))  # Original edge
        edge_list.append((v, u))  # Reverse edge to make it bidirectional
        
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index