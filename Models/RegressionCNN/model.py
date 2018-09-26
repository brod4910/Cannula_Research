import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

class Model(nn.Module):

    def __init__(self, feature_layers, classifier, checkpoint= False):
        super(Model, self).__init__()
        if feature_layers is not None:
            self.feature_layers = feature_layers
        else:
            self.feature_layers = None

        self.classifier = classifier
        self.checkpoint = checkpoint

    def forward(self, x):
        if self.checkpoint is True:
            input = x.detach()
            input.requires_grad = True
            input = checkpoint_sequential(self.feature_layers, 2, input)
        else:
            if self.feature_layers is not None:
                input = self.feature_layers(x)
            else:
                input = x

        input = input.view(input.size(0), -1)
        input = self.classifier(input)
        return input
