import os
import random
import torch
import numpy as np
import PIL.Image as Image

from torch.utils.data import Dataset
from torchvision import transforms, utils

class loadedDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.classes = sorted(os.listdir(self.root_dir))
		self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]
		self.acc_count = [self.count[0]]
		for i in range(1, len(self.count)):
				self.acc_count.append(self.acc_count[i-1] + self.count[i])
		# self.acc_count = [self.count[i] + self.acc_count[i-1] for i in range(1, len(self.count))]

	def __len__(self):
		l = np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))
		return l

	def __getitem__(self, idx):
		for i in range(len(self.acc_count)):
			if idx < self.acc_count[i]:
				label = i
				break

		class_path = self.root_dir + '/' + self.classes[label] 

		if label:
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
		else:
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

		_, file_name = os.path.split(file_path)

		frames = []
		file_list = sorted(os.listdir(file_path))

		for _, f in enumerate(file_list):
			frame = Image.open(file_path + '/' + f)
			frame = self.transform(frame)
			frames.append(frame)

		return frames, label, file_name
