import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset, DataLoader
from numpy.fft import fft2, fftshift
import cv2
import matplotlib.animation as animation

# Load the depth images
path = './dataset/SUSTech1K-Released-pkl/0000/00-nm/000/09-sync-000-LiDAR-PCDs_depths.pkl'
# path = "./dataset/SUSTech1K-Reorganized/key_points/1057_00-nm_270-far_11-sync-270-far-Camera-Pose.pkl"
# path = './dataset/SUSTech1K-Released-pkl/1057/00-nm/270-far/11-sync-270-far-Camera-Pose.pkl'
with open(path, 'rb') as f:
    depth_images = pickle.load(f)
print(depth_images[0,:,30,30])

# # convert the depth images into usually format
# depth_images = [depth_image.transpose(1, 2, 0) for depth_image in depth_images]
# depth_images = np.array([
#     cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
#     for depth_image in depth_images
# ])
# depth_images = np.asarray(depth_images)

# # play the depth sequence
# fig, ax = plt.subplots()
# img = ax.imshow(depth_images[0], cmap='gray')

# def update(frame):
#     img.set_data(frame)
#     return img,

# ani = animation.FuncAnimation(fig, update, frames=depth_images, interval=100)
# plt.show()

# # play only one depth image
# plt.imshow(depth_images[0], cmap='gray')
# plt.colorbar()
# plt.show()
