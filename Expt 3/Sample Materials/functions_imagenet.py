import numpy as np 
import pandas as pd
import subprocess
import os
import torchvision.models as models
import torchvision
import torch
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from xml.dom import minidom
from os.path import basename
from PIL import Image

from labels_imagenet import labels_dict


# Global variables
dataroot     = 'annon'
xml_folder   = dataroot + '/data/ILSVRC/Annotations/CLS-LOC/train/'
train_folder = dataroot + '/data/ILSVRC/Data/CLS-LOC/train/'
val_folder   = dataroot + '/data/ILSVRC/Data/CLS-LOC/val/'
IMG_SIZE     = 224
WORKERS      = 1
BATCH_SIZE   = 1


def get_labels_dict():
    labels = 'data/LOC_synset_mapping.txt'
    labels_dict = {}
    separator = ', '
    f = open(labels, "r")
    for l in f.readlines():
        split = l.split()
        index = split[0]
        labels = [word.strip() for word in ' '.join(split[1:]).split(',')]
        labels_dict[index] = separator.join(labels)
    return labels_dict


class ValDataset(Dataset):
    """Custom Dataset for loading val images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=1)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['label'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


def imagenet_dataloaders(transform_train=True):

	# Data loading code
	traindir = train_folder
	valdir = val_folder
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])


	if transform_train:
		train_dataset = datasets.ImageFolder(
			traindir,
				transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize
				]))

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=BATCH_SIZE, shuffle=True,
			num_workers=WORKERS, pin_memory=True, sampler=None)

	else:
		train_dataset = datasets.ImageFolder(
			traindir,
				transforms.Compose([
				transforms.Resize(224),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize
				]))

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=BATCH_SIZE, shuffle=False,
			num_workers=WORKERS, pin_memory=True, sampler=None)



	val_transform = transforms.Compose([transforms.Resize(224),
										transforms.CenterCrop(224),
										transforms.ToTensor(),
										normalize
										])

	val_dataset = ValDataset(    csv_path  = dataroot + '/val_data_labels.csv',
								 img_dir   = dataroot + '/data/ILSVRC/Data/CLS-LOC/val/',
								 transform = val_transform)

	val_loader = DataLoader( dataset=val_dataset,
							 batch_size=BATCH_SIZE,
							 shuffle=False,
							 num_workers=WORKERS)
	
	return train_loader, val_loader, train_dataset, val_dataset


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_loader_img(img, fam=False):
    img = deepcopy(img)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    input_tensor = unorm(img)
    
    if fam:
        plt.imshow(input_tensor[0].permute(1,2,0), alpha=1)
    else:
        plt.imshow(input_tensor[0].permute(1,2,0))


