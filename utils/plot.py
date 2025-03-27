# Plot loss curves
import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses, save_path):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss', marker='v')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    
def plot_single_skeleton(keypoints_pred, keypoint_sequence, save_path='./figure/demo_dct.png'):
    """Visualize predicted and ground truth skeletons."""
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

    # Plot the skeleton connections for ground truth and prediction
    for start, end in connections:
        ax.plot(
            [keypoint_sequence[start, 0], keypoint_sequence[end, 0]],
            [keypoint_sequence[start, 1], keypoint_sequence[end, 1]],
            'b-', label='Ground Truth' if start == 0 and end == 1 else ""
        )
        ax.plot(
            [keypoints_pred[start, 0], keypoints_pred[end, 0]],
            [keypoints_pred[start, 1], keypoints_pred[end, 1]],
            'r-', label='Predicted' if start == 0 and end == 1 else ""
        )

    # Plot each joint
    ax.scatter(keypoint_sequence[:, 0], keypoint_sequence[:, 1], c='b', marker='o', label='Ground Truth Joints')
    ax.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1], c='r', marker='x', label='Predicted Joints')

    # Set axis labels, aspect ratio, and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    # Save the plot
    plt.savefig(save_path)