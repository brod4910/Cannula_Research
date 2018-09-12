import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSE(nn.Module):
	def __init__(self):
		super(RMSE, self).__init__()

	def forward(self, output, target):
		return torch.sqrt(F.mse_loss(output, target))

def rmse_loss(output, target):
		return torch.sqrt(F.mse_loss(output, target))