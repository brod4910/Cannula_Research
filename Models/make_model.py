import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

class Model(nn.Module):

    def __init__(self, feature_layers, classifier, checkpoint= False):
        super(Model, self).__init__()
        self.checkpoint = checkpoint
        self.feature_layers = feature_layers
        self.classifier = classifier

    def forward(self, x):
        if self.checkpoint is True:
            input = x.detach()
            input.requires_grad = True
            input = checkpoint_sequential(self.feature_layers, 2, input)
        else:
            input = self.feature_layers(x)

        input = input.view(input.size(0), -1)
        input = self.classifier(input)
        return input

def make_layers(layout):
    layers = []

    for layer in layout:
        if layer[0] == 'A':
            layers += [nn.AvgPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'C':
            conv2d = nn.Conv2d(in_channels= layer[1], out_channels= layer[2], 
                kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], dilation= layer[5], padding= layer[6])
            if layer[6] == 'ReLU_NoB2d':
                layers += [conv2d, nn.ReLU(inplace= True)]
            elif layer[6] == 'ReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.ReLU(inplace= True)]
            elif layer[6] == 'PReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.PReLU(inplace= True)]
            elif layer[6] == 'SELU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.SELU(inplace= True)]
            elif layer[6] == 'LeakyReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.LeakyReLU(inplace= True)]
            elif layer[6] == 'Sigmoid':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.Sigmoid()]
            elif layer[6] == 'Sigmoid_NoB2d':
                layers += [conv2d, nn.Sigmoid()]
            elif layer[6] == 'Tanh':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.Tanh()]
            elif layer[6] == 'Tanh_NoB2d':
                layers += [conv2d, nn.Tanh()]
            else:
                layers += [conv2d]

    return nn.Sequential(*layers)

def make_classifier_layers(layout):
    layers = []
    for layer in layout:
        if layer[0] == 'L':
            if layer[3] == 'ReLU_NoB2d':
                layers += [nn.Linear(layer[1], layer[2]), nn.ReLU(inplace= True)]
            elif layer[3] == 'ReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.ReLU(inplace= True)]
            elif layer[3] == 'PReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.PReLU(inplace= True)]
            elif layer[3] == 'SELU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.SELU(inplace= True)]
            elif layer[3] == 'LeakyReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.LeakyReLU(inplace= True)]
            elif layer[0] == 'Tanh_NoB2d':
                layers += [nn.Linear(layer[1], layer[2]), nn.Tanh()]
        elif layer[0] == 'D':
            layers += [nn.Dropout(layer[1])]
        elif layer[0] == 'AD':
            layers+= [nn.AlphaDropout(layer[1])]
        elif layer[0] == 'FC':
            layers += [nn.Linear(layer[1], layer[2])]
        elif layer[0] == 'FC_Tanh':
            layers += [nn.Linear(layer[1], layer[2]), nn.Tanh()]

    return nn.Sequential(*layers)

