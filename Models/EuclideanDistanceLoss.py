import torch
import torch.nn as nn

class EuclideanDistanceLoss(nn.Module):
	def __init__(self):
		super(EuclideanDistanceLoss, self).__init__()

	def forward(self, output, target):
		return torch.sqrt(torch.sum((t2-t1)**2))

def ed_loss(output, target):
		return torch.sqrt(torch.sum((t2-t1)**2))

def ed_tensor(output, target):
	return torch.sqrt((t2-t1)**2)