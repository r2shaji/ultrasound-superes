import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.image_pairs = []

        # Gather all subdirectories under root_dir
        subdirs = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        # For each subdirectory, look for 'lr' and 'hr' folders
        for subdir in subdirs:
            lr_folder = os.path.join(root_dir, subdir, 'png_Low_Resolution')
            hr_folder = os.path.join(root_dir, subdir, 'png_High_Resolution')

            # If either folder doesn't exist, skip
            if not (os.path.exists(lr_folder) and os.path.exists(hr_folder)):
                continue

            # Collect all filenames in the LR folder
            lr_files = [
                f for f in os.listdir(lr_folder)
                if f.endswith('.png') or f.endswith('.jpg')
            ]

            hr_file = [
                f for f in os.listdir(hr_folder)
                if f.endswith('.png') or f.endswith('.jpg')
            ]

            for filename in lr_files:
                lr_path = os.path.join(lr_folder, filename)
                hr_path = os.path.join(hr_folder, hr_file[0])
                if os.path.exists(hr_path):
                    self.image_pairs.append((lr_path, hr_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.image_pairs[idx]
        filename = os.path.basename(lr_path)

        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)

        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0

        lr_img = np.expand_dims(lr_img, axis=0)
        hr_img = np.expand_dims(hr_img, axis=0)

        return torch.from_numpy(lr_img), torch.from_numpy(hr_img), lr_path