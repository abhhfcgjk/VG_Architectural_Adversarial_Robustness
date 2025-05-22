

import torch
import torchvision
import torchvision.transforms.functional as F
import folders


class DataLoader(object):
	"""
	Dataset class for IQA databases
	"""

	def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

		self.batch_size = batch_size
		self.istrain = istrain

		if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'clive')| (dataset == 'kadid10k'):
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'koniq':
			transforms = torchvision.transforms.Compose([
				torchvision.transforms.Resize((224, 224)),
				torchvision.transforms.ToTensor()])
		elif dataset == 'nips':
			transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((224, 224)),
					torchvision.transforms.ToTensor()])
		elif dataset == 'fblive':
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])


		if dataset == 'koniq':
			self.data = folders.Koniq_10kFolder(
				root=path, index=img_indx, transform=transforms, patch_num=1)
		elif dataset == 'nips':
			self.data = folders.NIPSFolder(
				root=path, index=img_indx, transform=transforms, patch_num=1)		

	def get_data(self):
		if self.istrain:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=self.batch_size, shuffle=True)
		else:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=1, shuffle=False)
		return dataloader