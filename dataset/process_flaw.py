import os
import pickle
import numpy as np
import shutil

# Paths to the directories containing the depth maps and key points
depth_dir = "./dataset/SUSTech1K-Reorganized/depth_map"
keypoint_dir = "./dataset/SUSTech1K-Reorganized/key_points"

# Directories to move flawed files to
flawed_depth_dir = "./dataset/SUSTech1K-Reorganized/flawed/depth_map"
flawed_keypoint_dir = "./dataset/SUSTech1K-Reorganized/flawed/key_points"

# # Create directories if they don't exist
# os.makedirs(flawed_depth_dir, exist_ok=True)
# os.makedirs(flawed_keypoint_dir, exist_ok=True)

# Walk through key_points directory and check each file
for root, _, files in os.walk(keypoint_dir):
    # print(root)
    for keypoint_file in files:
        keypoint_path = os.path.join(root, keypoint_file)
        with open(keypoint_path, 'rb') as f:
            keypoint_data = pickle.load(f)

            # Check if the shape of the key points data is (frames, 17, 3)
            if not (isinstance(keypoint_data, np.ndarray)
                    and keypoint_data.ndim == 3
                    and keypoint_data.shape[1:] == (17, 3)):

                # # If the shape is incorrect, move the flawed files
                # file_base = keypoint_file[:14]

                # # Move keypoint file
                # shutil.move(keypoint_path,
                #             os.path.join(flawed_keypoint_dir, keypoint_file))

                # # Find and move corresponding depth file
                # depth_file = next((f for f in os.listdir(depth_dir)
                #                    if f.startswith(file_base)), None)
                # if depth_file:
                #     shutil.move(os.path.join(depth_dir, depth_file),
                #                 os.path.join(flawed_depth_dir, depth_file))

                print(f"Moved flawed files: {keypoint_file}")
