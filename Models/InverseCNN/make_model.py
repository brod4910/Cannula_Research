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
		input = x.detach()
        input.requires_grad = True
        
		for idx, layer in enumerate(feature_layers.children()):
			if isinstance(layer, nn.Conv2d):
				input