import torch
import numpy as np
import torchvision
import random
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import pickle
import time
import scipy
import matplotlib.pyplot as plt
import torchvision.models as models

from PIL import Image
from copy import deepcopy
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.data import astronaut
from skimage.measure import label, regionprops
from skimage.color import label2rgb

from functions import *


DEVICE = 'cuda:0'
BATCH_SIZE = 12
IMG_SIZE = 224
UNIT = 32.
NUM_EPOCHS = 3
BETA = 1
ALPHA = 5  # for compatability only
MEAN_NORM=(0.485, 0.456, 0.406)
STD_NORM=(0.229, 0.224, 0.225)
FILTER_SIZE = 7
LEARNING_RATE = 0.001      # Default = 0.001
WEIGHT_DECAY  = 0.0001
MOMENTUM      = 0.9
WORKERS       = 2
USE_SP_BOX = False
SEGMENT_DIVISIONS = 30
PERC_USE = 0.3
TRAIN_ON_SEGMENT = True

transformNoNormalize = transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor()
])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

transformNormalize = transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize
])


def plot_loader_img(img, fam=False):
	img = deepcopy(img)
	
	if len(img.shape)==4:
		img = img[0]
	
	unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	input_tensor = unorm(img)

	if fam:
		plt.imshow(input_tensor.permute(1,2,0).cpu().detach().numpy(), alpha=1)
	else:
		plt.imshow(input_tensor.permute(1,2,0).cpu().detach().numpy())

	return input_tensor[0]


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


def get_cam(C, pred, cam_type='CAM'):
	"""
	Get a 7x7 CAM
	"""
	
	if len(C.shape) == 4:
		C = C[0]
		
	if cam_type == 'CAM':
		test_weights = WEIGHTS[pred] 
#         print(test_weights.shape, C.shape)
		C_conts = test_weights.reshape(test_weights.shape[0],1,1) * C
		return C_conts.sum(axis=0)
	
	if cam_type == 'FAM':
		gap = torch.nn.AvgPool2d(7)
		x = gap(C).view(-1)
		test_weights = WEIGHTS[pred]    
		c = x * test_weights
		nb_feature_idx = torch.argmax(c).item()
		cam = C[nb_feature_idx]  
		return cam
	
	if cam_type == 'Random':
		cam = torch.tensor(np.random.rand(7,7))
		return cam


def get_greedy_superpixel_ccrs():
	"""
	Get the important superpixel segments by order using greedy search
	"""
	
	image = (inverse_normalize(QUERY_IMG, mean=MEAN_NORM,
							   std=STD_NORM) * 255)[0].permute(1,2,
															   0).cpu().detach().numpy().round().astype('uint8')
	
	org_logits, _, _ = netC(QUERY_IMG)
	org_logit = org_logits[0][QUERY_PRED].item()    
	results = list()
	
	segments = slic(image, n_segments=SEGMENT_DIVISIONS, sigma=5, start_label=1)
	
	# Final list
	segs_used = list()
	final_results = list()
	
	for current_seg in range(len(regionprops(segments))+1):
		
		if current_seg == 0:
			pass
		else:
			segs_used.append(results[0][1])
			final_results.append(results[0])
						
		results = list()

		for idx, region in enumerate(regionprops(segments)):
			
			# Don't do previoulsy added segments
			if np.unique(segments)[idx] in segs_used:
				continue

			minr, minc, maxr, maxc = region.bbox
			
			occlude_sp = keep_ccr_superpixel(image, segments, np.unique(segments)[idx], segs_used) 
				
			PIL_image  = Image.fromarray(occlude_sp)
			torch_img  = transformNormalize(PIL_image).view(-1, 3, 224, 224)
			new_logits, _, _ = netC(torch_img.to(DEVICE))
			feature_logit = new_logits[0][QUERY_PRED].item()  
			new_pred = torch.argmax(new_logits, dim=1).item()
			
			occlude_sp = keep_ccr_superpixel(image, segments, np.unique(segments)[idx], []) 
			only_sp = crop_ccr_superpixel_box(occlude_sp, minr, minc, maxr, maxc, crop_image=True)
			PIL_image = Image.fromarray(only_sp)
			PIL_image = expand2square(PIL_image, background_color=0)
			torch_img = transformNormalize(PIL_image).view(-1, 3, 224, 224)
			_, x, _ = netC(torch_img.to(DEVICE))
			
			results.append([feature_logit, np.unique(segments)[idx], torch_img, x,  
									   new_pred,  
									   region, SEGMENT_DIVISIONS, segments ])
						
		results = sorted(results, key=lambda x: x[0], reverse=True) 

	return final_results


def inverse_normalize(in_tensor, mean=MEAN_NORM, std=STD_NORM):
	tensor = in_tensor.clone().detach()
	for t, m, s in zip(tensor, mean, std):
		t.mul_(s).add_(m)
	return tensor


def get_transformed_traning_data(idx, loader):
	"""
	Takes in indexs and a training dataloader
	returns: Those indexs transformed
	"""
	
	img = new_transform(  Image.open(train_dataset.imgs[idx][0]).convert('RGB')  ).view(-1,3,IMG_SIZE, IMG_SIZE)

	return img


def largest_indices(ary, n):
	"""
	Order 2d array by largest values
	"""
	
	flat = ary.flatten()
	indices = np.argpartition(flat, -n)[-n:]
	indices = indices[np.argsort(-flat[indices])]
	return np.unravel_index(indices, ary.shape)


def get_superpixel_ccrs(keep_sp=True):
	"""
	Carve out the superpixel precisely
	"""
	
	image = (inverse_normalize(QUERY_IMG,
							   mean=MEAN_NORM,
							   std=STD_NORM) * 255)[0].permute(1,2,
															   0).cpu().detach().numpy().round().astype('uint8')
	
	
	results = list()

	
	segments = slic(image, n_segments=SEGMENT_DIVISIONS, sigma=5, start_label=1)
	
	
	for idx, region in enumerate(regionprops(segments)):

		occlude_sp = keep_ccr_superpixel(image, segments, np.unique(segments)[idx], [], keep_sp=keep_sp) 
		PIL_image  = Image.fromarray(occlude_sp)
		torch_img  = transformNormalize(PIL_image).view(-1, 3, 224, 224)
		new_logits, _, _ = netC(torch_img.to(DEVICE))
		feature_logit = new_logits[0][QUERY_PRED].item()  
		results.append([feature_logit, idx+1, torch_img, None,  
								   None,  
								   region, SEGMENT_DIVISIONS, segments ])
		
	if keep_sp:
		results = sorted(results, key=lambda x: x[0], reverse=True) 
	
	if not keep_sp:
		results = sorted(results, key=lambda x: x[0], reverse=False) 
		
	return results


def crop_center(img, cropx, cropy):
	y, x, c = img.shape
	startx = x//2 - cropx//2
	starty = y//2 - cropy//2    
	return img[starty:starty+cropy, startx:startx+cropx, :]


def get_centre_cropped_image(query_idx, training=False):
	"""
	data index, training or not
	return: centre cropped
	"""
	
	if training:
		image = cv2.imread(  train_dataset.imgs[query_idx][0]  )
	else:
		image = cv2.imread(val_dataset.img_dir + val_dataset.img_names[query_idx])
		
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	h = image.shape[0]
	w = image.shape[1]
	crop = min(h, w)
	image = crop_center(image, crop, crop)
	return image


def keep_ccr_superpixel(img, segments, segVal, seg_list=list(), keep_sp=True):
	"""
	Keep only the sp area
	"""
	
	if keep_sp:
		mask = np.zeros(img.shape, dtype = "uint8")
		mask[segments == segVal] = 1

		for i in range(len(seg_list)):
			mask[segments == seg_list[i]] = 1

		arr = img * mask
		return arr
	
	else:
		mask = np.zeros(img.shape, dtype = "uint8")
		mask += 1
		mask[segments == segVal] = 0

		for i in range(len(seg_list)):
			mask[segments == seg_list[i]] = 0

		arr = img * mask
		return arr


def crop_ccr_superpixel_box(img, minr, minc, maxr, maxc, crop_image=False):
	"""
	Blackout recantalge superpixel area (i think)
	"""
	
	top_left = (minc, minr)
	to_right = maxc - minc
	down     = maxr - minr
	
	if not crop_image:
		mask = np.zeros(img.shape)
		mask[minr: minr+down, minc: minc+to_right, :] = 1
		arr = img * mask
		arr = arr.astype('uint8')
				
		return arr
	else:
		return img[minr: minr+down, minc: minc+to_right, :]


def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return pil_img
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result





def get_max_2d(a):
	maxindex = a.argmax()
	return np.unravel_index(a.argmax(), a.shape)


def crop_ccr_cam_box(img, x, y, h, w):
	
	m = torch.nn.Upsample(scale_factor=7)
	
	if len(img) == 3:
		img = img[int(y): int(y+h), int(x): int(x+w), :]
		img = m(img)
		return img
	else:
		img = img[:, :, int(y): int(y+h), int(x): int(x+w)]
		img = m(img)
		return img


def get_num_cam_pixels_to_occlude(cam_type='CAM'):
	"""
	return: num pixels CAM includes with alpha = 5
	"""
	
	pixel_cam = get_upsampled_cam_query(netC, cam_type=cam_type, upsample=True)
	threshold = pixel_cam.flatten().max() / ALPHA
	num_pixels = (pixel_cam >= threshold).flatten().sum()
	return num_pixels, pixel_cam


def get_upsampled_cam_query(netC, cam_type='CAM', upsample=True):
	"""
	For getting a pixel-level CAM
	"""
	
	temp_weights = WEIGHTS[QUERY_PRED]
	
	if cam_type=='FAM':
		c = QUERY_X * temp_weights
		nb_feature_idx = torch.argmax(c).item()
		cam = QUERY_C[nb_feature_idx].cpu().detach().numpy()  # really the FAM
		
	elif cam_type=='CAM':
		cam = get_cam(QUERY_C, QUERY_PRED).cpu().detach().numpy()
		
	if cam_type=='Random':
		cam = np.random.rand(7,7)
		
	if upsample:
		cam = scipy.ndimage.zoom(cam, (32, 32), order=3) 
		return cam
	else:
		return cam


def get_cam_image_masked(cam, threshold, occlude_pos=True):
	"""
	Take in cam and num of pixels to change
	return a mask
	"""
	
	if occlude_pos:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape)
		mask += 1.

		for i in range(len(idx)):
			x, y = idx[i], idy[i]
			mask[x][y] = 0.

		img = QUERY_IMG.clone().detach()
		img *= mask
		img[img==0] = -2.1179
	
	else:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape)

		for i in range(len(idx)):
			x, y = idx[i], idy[i]
			mask[x][y] = 1.

		img = QUERY_IMG.clone().detach()
		img *= mask
		img[img==0] = -2.1179
	
	return img


def save(occ_type, accs, iterations, epochs):
	df = pd.DataFrame()
	df['Accuracy']   = accs
	df['Dataset']    = 'CUB-200'
	df['Iterations'] = iterations
	df['Technique']  = occ_type
	df['Epoch'] = epochs
	df['BETA'] = BETA
	df['SLIC'] = SEGMENT_DIVISIONS
	df['PrecIncluded'] = sum(avg_size) / len(avg_size) 
	df.to_csv('data/FullBetaResults/Training_InclusionSPs_' + str(BETA) + str(occ_type) + str(SEGMENT_DIVISIONS) + '.csv')


def evaluate_test(test_loader, netC, test_dataset, DEVICE):
	total_correct = 0

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			imgs, labels = data
			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

			logits, _, _ = netC(imgs)
			preds = torch.argmax(logits, dim=1)

			total_correct += sum(preds == labels).item()

	return total_correct / test_dataset.data.shape[0] 


def return_occluded_imgs(train_on_segment, perc_use=0.5):
	"""
	Return Occluded data for training
	"""
		
	# Get num original CAM pixels
	cam_pix_num, pixel_cam = get_num_cam_pixels_to_occlude()
	
	sp_results = get_superpixel_ccrs()
	
	# Find best approximation of sp data to CAM
	total_sp_blacked_out, sp_torch_img = find_num_sp_to_use(sp_results, cam_pix_num,
															train_on_segment=train_on_segment,
														   perc_use=perc_use)
	
		
	# Get Random
	_, pixel_rand = get_num_cam_pixels_to_occlude(cam_type='Random')
	rand_torch_img = resize_cam_with_sp(pixel_rand, total_sp_blacked_out, train_on_segment)
		
	return sp_torch_img, rand_torch_img, total_sp_blacked_out


def find_num_sp_to_use(results, cam_pix_num, train_on_segment, perc_use):
	"""
	take cam pixel number, and find best approximation
	"""

	total_blacked_out = 0
	img = QUERY_IMG.clone().detach()
	
	if train_on_segment:
		img = torch.zeros(img.shape)
   

	max_sal = max([x[0] for x in results])
	threshold = max_sal / BETA


	for i in range(len(results)):


		if results[i][0] < threshold:
			break

		sp_idx = results[i][1]
		segments = results[i][-1]
		img = blackout_ccr_superpixel(img, segments, sp_idx, train_on_segment=train_on_segment)
		
		# Change to black background
		temp = img.clone()
		temp[temp==0] = -2.1179
		
		logits, _, _ = netC(temp.to(DEVICE))
		logit = logits[0][QUERY_PRED].item()
		pred = torch.argmax(logits, dim=1).item()
		total_blacked_out += (segments == sp_idx).sum()
		prob = torch.softmax(logits, dim=1)[0][QUERY_PRED].item() 
		

	if 'temp' not in locals():
		total_blacked_out = 224**2
		temp = torch.zeros(QUERY_IMG.shape)
		temp[temp==0] = -2.1179

	return total_blacked_out, temp


def blackout_ccr_superpixel(img, segments, segVal, train_on_segment=False):
	"""
	Choose to Keep only the sp area or not
	"""

	img = img.cpu().detach()
	
	if not train_on_segment:
		mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype="float32")
		mask += 1
		mask[segments == segVal] = 0
		mask = np.array([mask, mask, mask]).reshape(1, 3, IMG_SIZE, IMG_SIZE)
		mask = torch.tensor(mask)
		arr = img * mask
		
	else:
		mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype="float32")
		mask[segments == segVal] = 1
		mask = np.array([mask, mask, mask]).reshape(1, 3, IMG_SIZE, IMG_SIZE)
		mask = torch.tensor(mask)
		new_unmasked_area = QUERY_IMG.cpu() * mask
		arr = img + new_unmasked_area
	
	return arr


# In[30]:


def resize_cam_with_sp(cam, threshold, train_on_segment):
	
	if not train_on_segment:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape)
		mask += 1.
		mask[idx, idy] = 0

		img = QUERY_IMG.clone().cpu().detach()
		img *= mask
		img[img==0] = -2.1179

		return img

	else:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape)
		mask[idx, idy] = 1

		img = QUERY_IMG.clone().cpu().detach()
		img *= mask
		img[img==0] = -2.1179

		return img


for SEGMENT_DIVISIONS in [10, 20, 30]:
	for BETA in [1, 1.05, 1.1, 1.25, 1.5, 2, 3]:

		print('=====================================================')
		print("BETA:", BETA, " -- SLIC:", SEGMENT_DIVISIONS)
		print('=====================================================')

		#### Training Loop

		torch.cuda.empty_cache()

		netC, WEIGHTS = collect_model(DEVICE)

		netC_retrain_sp, _ = collect_model(DEVICE)  # for retraining
		netC_retrain_rand, _ = collect_model(DEVICE)

		netC = netC.eval()

		netC_retrain_sp = netC_retrain_sp.train()
		netC_retrain_rand = netC_retrain_rand.train()

		train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(shuffle=False,
																				 training_transform=False,
																				 b_size=BATCH_SIZE)

		acc = evaluate_test(test_loader, netC, test_dataset, DEVICE)
		cce_loss = torch.nn.CrossEntropyLoss()

		optimizer_sp = torch.optim.SGD(netC_retrain_sp.parameters(), lr=LEARNING_RATE,
									   momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
		optimizer_rand = torch.optim.SGD(netC_retrain_rand.parameters(), lr=LEARNING_RATE,
										 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

		train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(shuffle=True,
																				 training_transform=True,
																				 b_size=BATCH_SIZE)

		avg_size = list()
		iterations = [0]
		epochs = [0]
		current_iter = 0


		start_time = time.time()
		current_iter = 0

		sp_accs = [acc]
		rand_accs = [acc]

		for epoch in range(NUM_EPOCHS):
			for i, data in enumerate(train_loader):

				row = pd.DataFrame()

				current_iter += 1

				netC_retrain_sp.zero_grad()
				netC_retrain_sp.train()
				netC_retrain_rand.zero_grad()
				netC_retrain_rand.train()

				imgs, labels = data
				imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

				sp_torch_imgs = torch.zeros(imgs.shape)
				rand_torch_imgs = torch.zeros(imgs.shape)

				with torch.no_grad():

					rolling_size = list()

					logits, imgs_x, imgs_C = netC(imgs)
					preds = torch.argmax(logits, dim=1)
					for k in range(len(imgs)):
						QUERY_LABEL = labels[k]
						QUERY_IMG = imgs[k].view(1, 3, IMG_SIZE, IMG_SIZE)
						QUERY_PRED = preds[k]
						QUERY_X = imgs_x[k]
						QUERY_C = imgs_C[k]

						sp_torch_img, rand_torch_img, a_size = return_occluded_imgs(
							train_on_segment=TRAIN_ON_SEGMENT, perc_use=PERC_USE)

						sp_torch_imgs[k] = sp_torch_img[0]
						rand_torch_imgs[k] = rand_torch_img[0]

						# convert from pixels to % of imag
						a_size /= 224**2

						rolling_size.append( a_size ) 

				logits, _, _ = netC_retrain_sp(sp_torch_imgs.to(DEVICE))
				classify_loss = cce_loss(logits, labels)
				classify_loss.backward()
				optimizer_sp.step()

				logits, _, _ = netC_retrain_rand(rand_torch_imgs.to(DEVICE))
				classify_loss = cce_loss(logits, labels)
				classify_loss.backward()
				optimizer_rand.step()

				if current_iter % 50 == 0:
					acc_sp = evaluate_test(test_loader, netC_retrain_sp.eval(), test_dataset, DEVICE)
					acc_rand = evaluate_test(test_loader, netC_retrain_rand.eval(), test_dataset, DEVICE)
					print("Meta Data:", current_iter, acc_sp, acc_rand)
					
					sp_accs.append(acc_sp)
					rand_accs.append(acc_rand)
					
					iterations.append(current_iter)
					epochs.append(epoch)
					avg_size.append( sum(rolling_size) / len(rolling_size)  )

					save('Superpixels', sp_accs, iterations, epochs)
					save('Random', rand_accs, iterations, epochs)
					
					print([acc_sp, acc_rand])
					print(  "Average Size:", sum(avg_size) / len(avg_size) )
					print(" ")
					
		print("Time Taken:", time.time() - start_time)
		print(  "Average Size:", sum(avg_size) / len(avg_size) )

