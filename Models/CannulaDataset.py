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
    def __init__(self, input_file, target_file, root_dir, transform= None):
        self.root_dir = root_dir
        self.inputs = np.load(os.path.join(root_dir, input_file))
        self.inputs = np.expand_dims(self.inputs, 3)
        self.targets = np.load(os.path.join(root_dir, target_file))
        self.transform = transform
        kfold = kFold(inputs, targets)


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


def kFold(inputs, targets):
        kfold = KFold(5, True, 11)
        idxs = []

        for train, test in enumerate(kfold.split(self.inputs, self.targets)):
            idxs.append([train, test])

        return idxs


