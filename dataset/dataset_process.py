import os
import shutil

# Define the paths and file patterns
base_dir = "./SUSTech1K-Released-pkl"
depth_dir = "./SUSTech1K-Reorganized/depth_map"
keypoint_dir = "./SUSTech1K-Reorganized/key_points"

# required_files = [
#     "08-sync-{}-far-LiDAR-PCDs.pkl",
#     "09-sync-{}-far-LiDAR-PCDs_depths.pkl",
#     "10-sync-{}-far-LiDAR-PCDs_sils.pkl",
#     "11-sync-{}-far-Camera-Pose.pkl",
#     "12-sync-{}-far-Camera-Ratios-HW.pkl",
#     "13-sync-{}-far-Camera-RGB_raw.pkl",
#     "14-sync-{}-far-Camera-Sils_aligned.pkl",
#     "15-sync-{}-far-Camera-Sils_raw.pkl"
# ]

# Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


# Function to extract and move the required files
def extract_files(base_dir):
    count_a = 0
    count_b = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("09"):
                newFileName = root[2:] + '_' + file
                newFileName = "_".join(newFileName.split('/')[1:])
                print(newFileName)
                count_a += 1
                shutil.copyfile(os.path.join(root, file),
                                os.path.join(depth_dir, newFileName))
            elif file.startswith("11"):
                newFileName = root[2:] + '_' + file
                newFileName = "_".join(newFileName.split('/')[1:])
                print(newFileName)
                count_b += 1
                shutil.copyfile(os.path.join(root, file),
                                os.path.join(keypoint_dir, newFileName))
    print(f"depth sequence number: {count_a}")
    print(f"pose sequence: {count_b}")


# Run the function
# extract_files(base_dir)

print("Reorganization completed!")

# for root, dirs, files in os.walk(base_dir):
#     a = 0
#     if a == 1:
#         break
#     for file in files:
#         if file.startswith("09"):
#             a = 1
#             newFileName = root[2:] + '_' + file
#             newFileName = "_".join(newFileName.split('/')[1:])
#             print(newFileName)
#             shutil.copyfile(os.path.join(root, file),
#                             os.path.join(depth_dir, newFileName))
