import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
from numpy.fft import fft2, fftshift
import cv2


# Function to compute the FFT of an image
def compute_fft(image):
    return fftshift(fft2(image))


def fft_block(depth_images):
    # convert the depth images into usually format
    depth_images = [
        depth_image.transpose(1, 2, 0) for depth_image in depth_images
    ]
    depth_images = np.array([
        cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        for depth_image in depth_images
    ])

    # Compute FFTs of all depth maps
    fft_maps = np.array([compute_fft(depth_map) for depth_map in depth_images])

    # Calculate frequency changes between successive depth maps
    frequency_changes = []
    fft_map_len = len(fft_maps)
    for i in range(1, fft_map_len):
        freq_change = np.abs(fft_maps[i] - fft_maps[i - 1])
        frequency_changes.append(freq_change)
    frequency_changes.append(np.abs(fft_maps[fft_map_len - 1]))

    # Convert to a NumPy array for easier manipulation
    frequency_changes = np.array(frequency_changes)
    frequency_changes = np.expand_dims(frequency_changes, 1)
    return frequency_changes
    # print(frequency_changes.shape)


if __name__ == '__main__':
    # # Load the depth images
    # path = './dataset/SUSTech1K-Released-pkl/0000/00-nm/000/11-sync-000-Camera-Pose.pkl'
    path = './dataset/SUSTech1K-Released-pkl/0000/00-nm/000/09-sync-000-LiDAR-PCDs_depths.pkl'
    with open(path, 'rb') as f:
        depth_images = pickle.load(f)
    print(depth_images.shape)
    a = fft_block(depth_images)
    print(a.shape)
    # # Visualize frequency change for a specific pair of depth maps
    # index = 0  # Choose the pair to visualize the change
    # plt.imshow(np.log(np.abs(frequency_changes[index]) + 1), cmap='gray')
    # plt.colorbar()
    # plt.title(f'Frequency Change Between Depth Map {index} and {index + 1}')
    # plt.show()

    # play the depth sequence
    # fig, ax = plt.subplots()
    # img = ax.imshow(frequency_changes[0])

    # def update(frame):
    #     img.set_data(frame)
    #     return img,

    # ani = animation.FuncAnimation(fig,
    #                               update,
    #                               frames=frequency_changes,
    #                               interval=100)
    # plt.show()
