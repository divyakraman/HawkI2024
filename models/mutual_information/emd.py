# https://discuss.pytorch.org/t/implementation-of-squared-earth-movers-distance-loss-function-for-ordinal-scale/107927/3

import os
import numpy as np 

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

class EMD(nn.Module):

	def __init__(self, sigma=0.1, num_bins=256, normalize=True):
		super(EMD, self).__init__()

		self.sigma = sigma
		self.num_bins = num_bins
		self.normalize = normalize
		self.epsilon = 1e-10

		self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)


	def marginalPdf(self, values):
		residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def getEMD(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''

		# Torch tensors for images between (0, 1)
		input1 = input1*255
		input2 = input2*255

		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)

		emd = torch.mean(torch.square(torch.cumsum(pdf_x1, dim=-1) - torch.cumsum(pdf_x2, dim=-1)), dim=-1)

		return emd


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''
		return self.getEMD(input1, input2)


