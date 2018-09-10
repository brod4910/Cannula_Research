from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class CannulaDataset(Dataset):
    def __init__(self, input_file, target_file, root_dir, data_length, transform= None):
        self.inputs = np.load(os.path.join(root_dir, input_file))
        self.targets = np.load(os.path.join(root_dir, target_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        image = self.inputs[idx]
        label = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        image = image.type(torch.FloatTensor)

        return image, label
