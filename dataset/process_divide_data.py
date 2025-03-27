import os
import shutil

# Define the paths and file patterns
# base_dir = "./SUSTech1K-Released-pkl"
depth_source = "./SUSTech1K-Reorganized/depth_map"
keypoint_source = "./SUSTech1K-Reorganized/key_points"
depth_nm = "./SUSTech1K-Reorganized-divided/depth_map_nm"
keypoint_nm = "./SUSTech1K-Reorganized-divided/key_points_nm"
depth_obscure = "./SUSTech1K-Reorganized-divided/depth_map_obscure"
keypoint_obscure = "./SUSTech1K-Reorganized-divided/key_points_obscure"

# Function to extract and move the required files
def extract_files(base_dir,dest_nm,dest_obscure):
    count_a = 0
    count_b = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            segments = file.split("-")
            # print(len(segments))
            if segments[0].endswith('00'):
                count_a += 1
                shutil.copyfile(os.path.join(root, file),
                                os.path.join(dest_nm, file))
            else:
                count_b += 1
                shutil.copyfile(os.path.join(root, file),
                                os.path.join(dest_obscure, file))
    print(f"depth sequence number: {count_a}")
    print(f"pose sequence: {count_b}")


# Run the function
# extract_files(depth_source,depth_nm,depth_obscure)
extract_files(keypoint_source,keypoint_nm,keypoint_obscure)

print("Reorganization completed!")
