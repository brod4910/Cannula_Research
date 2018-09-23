from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from sklearn.model_selection import KFold

[self.inputs = np.load(os.path.join(root_dir, input_file))
self.inputs = np.expand_dims(self.inputs, 3)
self.targets = np.load(os.path.join(root_dir, target_file))
self.root_dir = root_dir]

kfold = KFold(4, True, 11)

for train, test in kfold.split(new_data):
    print('train: %s, test: %s' % (new_data[train], new_data[test]))
	
print(train, test)
