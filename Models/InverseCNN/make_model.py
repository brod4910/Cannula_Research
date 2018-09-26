import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

# U-shape CNN model maker

class Model(nn.Module):

	# feature layers describe the first half of the forward process
	# inverse_layers describe the second half of the forward process where image 
	# reconstruction happens
	def __init__(self, feature_layers, inverse_layers):
		self.feature_layers = feature_layers
		self.inverse_layers = inverse_layers

	def forward(self, x):
		filters = []
		pooling_modules = []
		input = x.detach()
        input.requires_grad = True

        # enumerate over the layers and save any convolutional layer's 
        # output and maxpooling output
		for __, layer in enumerate(self.feature_layers.children()):
			input = layer(input)

			if isinstance(layer, nn.Conv2d):
				filters += input
			if isinstance(layer, nn.MaxPool2d):
				pooling_modules += input

		# enumerate over the layers and use saved filters plus previous layers output as
		# input to next convolutional layer. The filters are connected to its respective
		# convolutional layer in a U-shaped fashion
		for __, layer in enumerate(self.inverse_layers.children()):
			if isinstance(layer, nn.Conv2d):
				input = layer(torch.cat([input, filters.pop(-1)], 1))
			else:
				input = layer(input)

		return input


