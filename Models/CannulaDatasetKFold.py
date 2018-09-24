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
    def __init__(self, inputs, targets, kfold, transform= None):
        self.root_dir = root_dir
        self.inputs = inputs[kfold]
        self.targets = targets[kfold]
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        image = self.inputs[idx]
        label = self.targets[idx]

        image = np.transpose(image, (2,1,0))
        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label





