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
import time
import random

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from xml.dom import minidom
from os.path import basename
from PIL import Image


dataroot      = '/home/people/16206131/scratch'
train_folder  = dataroot + '/data/ILSVRC/Data/CLS-LOC/train/'
val_folder    = dataroot + '/data/ILSVRC/Data/CLS-LOC/val/'
IMG_SIZE      = 224
WORKERS       = 4
BATCH_SIZE    = 256
DEVICE        = 'cuda'
NUM_EPOCHS    = 1
LEARNING_RATE = 0.001     
WEIGHT_DECAY  = 0.0001
MOMENTUM      = 0.9
ALPHA         = None


class ValDataset(Dataset):
	"""Custom Dataset for loading CelebA face images"""

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


class CustomTransform(object):

	def __call__(self, sample):
		image = sample
		unit = 32
		chance = np.random.randint(0, 2)
		if chance == 0:
			i = np.random.randint(0, 7)
			j = np.random.randint(0, 7)
			image[:, i*unit: i*unit+unit, j*unit: j*unit+unit] = 0.0
		return image


def imagenet_dataloaders(transform_train=True):

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
				transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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


class NetC(torch.nn.Module):
	def __init__(self, net):
		super(NetC, self).__init__()
		self.main = torch.nn.Sequential(*list(net.children()))[:-2]
		self.avgp = net.avgpool
		self.linear = net.fc

	def forward(self, x):
		C = self.main(x)
		x = self.avgp(C)
		x = x.view(-1, 2048)
		logits = self.linear(x)
		return logits, C


def collect_model():
	resnet = models.resnet50(pretrained=True).train()
	netC = NetC(resnet)
	weights = resnet.fc.weight
	netC.to(DEVICE)
	return netC, weights


def blackout_img(img, blackout_segs):
	unit = 32.
	filter_size = 7
	for coords in blackout_segs:
		i, j = coords
		img[:, int(i*unit): int(i*unit+unit), int(j*unit): int(j*unit+unit)] = -2.11
	return img


def largest_indices(ary, n):
	"""Returns the n largest indices from a numpy array."""
	flat = ary.flatten()
	indices = np.argpartition(flat, -n)[-n:]
	indices = indices[np.argsort(-flat[indices])]
	return np.unravel_index(indices, ary.shape)


def get_img_nb_coords(C, pred, weights):
	
	cam = get_cam(C, pred, weights).cpu().detach().numpy()
	fam = get_fam(C, pred, weights).cpu().detach().numpy()
	
	
	num_ccrs = get_num_ccrs(cam)
	
	ccr_indices = largest_indices(cam, 49)
	fam_indices = largest_indices(fam, 49)

	rand_indicies = get_blackout_coords(49 - num_ccrs)
		
	
	fam_results = list()
	for c in zip(fam_indices[0][num_ccrs:], fam_indices[1][num_ccrs:]):
		fam_results.append([c[0], c[1]])
	
	cam_results = list()
	for c in zip(ccr_indices[0][num_ccrs:], ccr_indices[1][num_ccrs:]):
		cam_results.append([c[0], c[1]])

	return cam_results, fam_results, rand_indicies, 49 - num_ccrs


def get_blackout_coords(blackout_segs):

	rand_blackout_segs = list()
	while len(rand_blackout_segs) < blackout_segs:
		i, j = random.randint(0, 6), random.randint(0, 6)
		if [i, j] not in rand_blackout_segs:
			rand_blackout_segs.append([i, j])
			
	return rand_blackout_segs


def get_num_ccrs(cam):
	results = list()
	max_saliency = cam.max()
	threshold = max_saliency / ALPHA
	return sum(cam.flatten() >= threshold)




def get_cam(C, pred, weights):
	test_weights = weights[pred.item()]
	C_conts = test_weights.reshape(test_weights.shape[0],1,1) * C
	return C_conts.sum(axis=0)



def evaluate_validation(netC, val_loader, val_dataset):


	with torch.no_grad():

		top1_correct = 0
		top5_correct = 0

		for i, data in enumerate(val_loader):

			imgs, labels = data
			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
			
			logits, _ = netC(imgs)
			preds = torch.argmax(logits, axis=1)
			top1_correct += torch.sum(preds == labels)

			_, top_5_preds = logits.topk(5)
			top5_correct += (labels.reshape(imgs.shape[0], 1).expand(imgs.shape[0], 5) == top_5_preds).sum()
				
	return top1_correct.item() / val_dataset.y.size



def get_fam(C, pred, weights):
	"""
	Get FAM in C
	return: 2D array of FAM
	"""
	
	gap = torch.nn.AvgPool2d(7)
	x = gap(C).view(2048)
	c = x * weights[pred]
	fam_idx = torch.argmax(c)
	fam = C[fam_idx]
	
	return fam


def save(occ_type, accs, iterations, epochs):
	df = pd.DataFrame()
	df['Accuracy']     = accs
	df['Dataset']      = 'ImageNet'
	df['Iterations']   = iterations
	df['Technique']    = occ_type
	df['Epoch']        = epochs
	df['ALPHA']        = ALPHA
	df['PrecIncluded'] = sum(avg_size) / len(avg_size)
	df.to_csv('data/Training_ALPHA_' + occ_type + '_' + str(ALPHA) + '.csv')


for ALPHA in [1, 1.05, 1.1, 1.25, 1.5, 2, 3]:

	print(" ")
	print(" ")
	print(" =======================================  ")
	print("ALPHA:", ALPHA)
	print(" =======================================  ")
	print(" ")


	#### Training Loop

	torch.cuda.empty_cache()

	netC, WEIGHTS = collect_model()    # for calculating occlusions

	netC_retrain_cam, _  = collect_model()
	netC_retrain_fam, _  = collect_model()
	netC_retrain_rand, _ = collect_model()

	netC = netC.eval()

	netC_retrain_cam = netC_retrain_cam.train()
	netC_retrain_fam = netC_retrain_fam.train()
	netC_retrain_rand = netC_retrain_rand.train()

	train_loader, test_loader, train_dataset, test_dataset = imagenet_dataloaders(transform_train=False)
	acc = evaluate_validation(netC, test_loader, test_dataset)
	cce_loss = torch.nn.CrossEntropyLoss()

	optimizer_cam = torch.optim.SGD(netC_retrain_cam.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
	optimizer_fam = torch.optim.SGD(netC_retrain_fam.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
	optimizer_rand = torch.optim.SGD(netC_retrain_rand.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

	print("Start Training...")
	train_loader, test_loader, train_dataset, test_dataset = imagenet_dataloaders(transform_train=True)

	avg_size = list()
	iterations = [0]
	epochs = [0]
	current_iter = 0

	start_time = time.time()
	current_iter = 0

	cam_accs = [acc]
	fam_accs = [acc]
	rand_accs = [acc]


	start_time = time.time()

	for i, data in enumerate(train_loader):
		
		current_iter += 1

		netC_retrain_cam.zero_grad()
		netC_retrain_cam.train()
		netC_retrain_fam.zero_grad()
		netC_retrain_fam.train()
		netC_retrain_rand.zero_grad()
		netC_retrain_rand.train()

		imgs, labels = data
		imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

		cam_torch_imgs  = torch.zeros(imgs.shape)
		fam_torch_imgs  = torch.zeros(imgs.shape)
		rand_torch_imgs = torch.zeros(imgs.shape)


		with torch.no_grad():

			rolling_size = list()

			logits, Cs = netC(imgs)
			preds = torch.argmax(logits, dim=1)

			for k in range(len(imgs)):

				cam_coords, fam_coords, rand_coords, size = get_img_nb_coords(Cs[k], preds[k], WEIGHTS)

				cam_torch_imgs[k]  = blackout_img(imgs[k].clone().detach(), cam_coords)
				fam_torch_imgs[k]  = blackout_img(imgs[k].clone().detach(), fam_coords)
				rand_torch_imgs[k] = blackout_img(imgs[k].clone().detach(), rand_coords)
				rolling_size.append(size) 
				

		logits, _ = netC_retrain_cam(cam_torch_imgs.to(DEVICE))
		classify_loss = cce_loss(logits, labels)
		classify_loss.backward()
		optimizer_cam.step()

		logits, _ = netC_retrain_fam(fam_torch_imgs.to(DEVICE))
		classify_loss = cce_loss(logits, labels)
		classify_loss.backward()
		optimizer_fam.step()

		logits, _ = netC_retrain_rand(rand_torch_imgs.to(DEVICE))
		classify_loss = cce_loss(logits, labels)
		classify_loss.backward()
		optimizer_rand.step()


		if current_iter % 50 == 0:
			acc_cam  = evaluate_validation(netC_retrain_cam.eval(),  test_loader, test_dataset)
			acc_fam  = evaluate_validation(netC_retrain_fam.eval(),  test_loader, test_dataset)
			acc_rand = evaluate_validation(netC_retrain_rand.eval(), test_loader, test_dataset)
			print("Meta Data:", current_iter, acc_cam, acc_fam, acc_rand)
			
			cam_accs.append(acc_cam)
			fam_accs.append(acc_fam)
			rand_accs.append(acc_rand)
			
			iterations.append(current_iter)
			epochs.append(1)
			avg_size.append( sum(rolling_size) / len(rolling_size)  )

			save('CAM', cam_accs, iterations, epochs)
			save('FAM', fam_accs, iterations, epochs)
			save('Random', rand_accs, iterations, epochs)

			print("Average Size:", sum(avg_size) / len(avg_size) )
			print(" ")
			


