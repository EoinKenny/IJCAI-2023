import torch
import numpy as np
import torchvision
import random
import pandas as pd
import argparse
import cv2
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
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.data import astronaut
from skimage.measure import label, regionprops
from skimage.color import label2rgb

from functions import *


DEVICE = 'cuda:1'
BATCH_SIZE = 1
NUM_WORKERS = 2
IMG_SIZE = 224
UNIT = 32.
BLACKOUT_NUM = 1
ALPHA = 5
# BETA = np.float('inf')
FILTER_SIZE = 7

MEAN_NORM=(0.485, 0.456, 0.406)
STD_NORM=(0.229, 0.224, 0.225)

SEGMENT_DIVISIONS = 30
USE_SP_BOX = False


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



train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(shuffle=False,
																		 training_transform=False,
																		 b_size=BATCH_SIZE)


DATAROOT = 'data'



X_train_c = np.load(DATAROOT + "/X_train_cont.npy")
X_test_c = np.load(DATAROOT + "/X_test_cont.npy")
X_train_x = np.load(DATAROOT + "/X_train_x.npy")
X_test_x = np.load(DATAROOT + "/X_test_x.npy")
train_preds = np.load(DATAROOT + "/X_train_y.npy")
test_preds = np.load(DATAROOT + "/X_test_y.npy")


# X_train = train_loader.dataset.data
y_train = train_loader.dataset.data.target.values

# X_test = test_loader.dataset.data
y_test = test_loader.dataset.data.target.values



# Fit COLE and DkNN
twin = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric='euclidean') 
twin.fit(X_train_c, train_preds)



def evaluate_validation(netC, test_loader):
	with torch.no_grad():
		top1_correct = 0
		top5_correct = 0
		for i, data in enumerate(test_loader):
			imgs, labels = data
			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
			logits, _, _ = netC(imgs)
			preds = torch.argmax(logits, axis=1)
			top1_correct += torch.sum(preds == labels)
			_, top_5_preds = logits.topk(5)
			top5_correct += (labels.reshape(imgs.shape[0], 1).expand(imgs.shape[0], 5) == top_5_preds).sum()

	print( "\n Validation Accuracy Top 1: " + str(top1_correct.item() / test_loader.dataset.data.target.shape[0] ) )
	print( "\n Validation Accuracy Top 5: " + str(top5_correct.item() / test_loader.dataset.data.target.shape[0] ) )



def get_cam(C, pred, cam_type='CAM'):
	
	if len(C.shape) == 4:
		C = C[0]
		
	if cam_type == 'CAM':
		test_weights = WEIGHTS[pred] 
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


def get_salient_regions(Conv, org_logits, net_classifier, cam_type='CAM'):

	query_pred = torch.argmax(org_logits, dim=1).item()

	if cam_type == 'Random':
		cam_type == 'CAM'

	
	q_cam = get_cam(Conv, query_pred, cam_type=cam_type)   
	
#     q_i, q_j = get_ccr_in_query(q_cam.cpu().detach().numpy())
	q_i, q_j = get_max_2d(q_cam.cpu().detach().numpy())
	
	q_feature = Conv[:, :, q_i:q_i+1, q_j:q_j+1 ]
	q_saliency = q_cam[q_i][q_j]

	# For random
	rand_int1 = random.randint(0, 5)
	rand_int2 = random.randint(0, 5)
	r_feature = Conv[:, :, rand_int1:rand_int1+1, rand_int2:rand_int2+1 ]
	r_saliency = q_cam[rand_int1][rand_int2]

	return [[q_saliency, [q_i, q_j], q_feature], [r_saliency, [rand_int1, rand_int2], r_feature]]



def crop_center(img, cropx, cropy):
	y, x, c = img.shape
	startx = x//2 - cropx//2
	starty = y//2 - cropy//2    
	return img[starty:starty+cropy, startx:startx+cropx, :]


def keep_ccr_superpixel(img, segments, segVal, seg_list=list()):
	"""
	Keep only the sp area
	"""
	
	mask = np.zeros(img.shape, dtype = "uint8")
	mask[segments == segVal] = 1
	
	for i in range(len(seg_list)):
		mask[segments == seg_list[i]] = 1
		
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


def get_nn_superpixel_ccrs(nn_idx):
	"""
	Carve out the superpixel precisely
	"""

	
	nn_torch_img = train_loader.dataset[nn_idx][0].view(1, 3, 224, 224)
	image = (inverse_normalize(nn_torch_img, mean=MEAN_NORM, std=STD_NORM) * 255)[0].permute(1,2,0).cpu().detach().numpy().round().astype('uint8')

	results = list()

	segments = slic(image, n_segments=SEGMENT_DIVISIONS, sigma=5, start_label=1)

	for idx, region in enumerate(regionprops(segments)):

		minr, minc, maxr, maxc = region.bbox

		# For logit
		occlude_sp = keep_ccr_superpixel(image, segments, np.unique(segments)[idx], []) 
		PIL_image = Image.fromarray(occlude_sp)
		torch_img = transformNormalize(PIL_image).view(-1, 3, 224, 224)
		logits, _, _ = netC(torch_img.to(DEVICE))
		saliency = logits[0][QUERY_PRED]

		# For x
		occlude_sp = crop_ccr_superpixel_box(occlude_sp, minr, minc, maxr, maxc, crop_image=True)
		PIL_image = Image.fromarray(occlude_sp)
		PIL_image = expand2square(PIL_image, background_color=0)
		torch_img = transformNormalize(PIL_image).view(-1, 3, 224, 224)
		_, x, _ = netC(torch_img.to(DEVICE))

		results.append([saliency, idx+1, torch_img, x, None, region, SEGMENT_DIVISIONS, segments ])

	return results


def get_coords_nb_feature_in_nn(Conv, cam, query_feature):

	# Threshold for saliency
	threshold = cam.max() / ALPHA
	coords = None
	max_dist = float('inf')
	for i in range(Conv.shape[2]):
		for j in range(Conv.shape[3]):
			temp_feature = Conv[:, :, i:i+1, j:j+1 ]
			dist = torch.cdist(query_feature.view(-1, 512), temp_feature.view(-1, 512), p=2.0).item() 
			
			if cam[i][j] > threshold:
				if dist < max_dist:
					max_dist = dist
					coords = [i, j]
					
	return max_dist, coords


def get_max_2d(a):
	maxindex = a.argmax()
	return np.unravel_index(a.argmax(), a.shape)


def get_ccr_in_query(cam):
	threshold = cam.flatten().max() / ALPHA
	unit = 32.
	data = list()
	for y in range(7):
		for x in range(7):
			if cam[y][x] > threshold:
				mask = torch.zeros(QUERY_IMG.shape)
				mask[:, :, int(y*unit): int(y*unit+unit), int(x*unit): int(x*unit+unit)] = 1
				mask = mask.to(DEVICE)
				ccr = QUERY_IMG * mask
				logits, _, _ = netC(ccr)
				ccr_logit = logits[0][QUERY_PRED].item()
				data.append([ccr_logit, [y, x,]])
			
	data = sorted(data, key=lambda x: x[0], reverse=True) 
	return data[0][1]


def get_sp_ccr_dist(QUERY_IMG, netC, results):

	ccr = results[0][3]  # latent representation   
	match = [None, None]
	min_l2 = float('inf')

	for nn_idx in NEIGHBORS:

		nn_results = get_nn_superpixel_ccrs(nn_idx)
		xs = [x[3] for x in nn_results]
		max_sal = max([x[0] for x in nn_results])
		threshold = 0.

		for i, x in enumerate(xs):
			if nn_results[i][0] >= threshold:
				dist = torch.cdist(x, ccr).item()
				if dist < min_l2:
					min_l2 = dist
					match = [nn_idx, nn_results[i]] 

	nn_idx = match[0]
	nn_ccr_idx = match[1][1]  # idx of superpixel
	nn_segs = match[1][-1]
	nn_ccr_pix = match[1][2]

	return torch.cdist(results[0][3], match[1][3])


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


def get_cam_ccr_dist(torch_img, netC, cam_type='CAM'):

	# Get Query CCR
	torch_img = QUERY_IMG
	unit = IMG_SIZE / FILTER_SIZE
	original_logits, original_x, original_C = netC(torch_img.to(DEVICE))
	query_nb_boxes = get_salient_regions(original_C,
										 original_logits,
										 net_classifier,
										cam_type=cam_type)  

	coord = query_nb_boxes[0][1]

	# print("Real coords:", coord)

	x = coord[1] * unit
	y = coord[0] * unit
	h, w = unit, unit

	ccr_pixels = crop_ccr_cam_box(torch_img, x, y, h, w)

	_, query_x, _ = netC(ccr_pixels.to(DEVICE))

	# Get Neighbor CCR
	results = list()

	query_feature = query_nb_boxes[0][2]

	for nn_idx in range(len(NEIGHBORS)):
		nn_img = train_loader.dataset[NEIGHBORS[nn_idx]][0].view(1, 3, 224, 224)
		nn_logits, nn_x, nn_C = netC(nn_img.to(DEVICE))
		nn_pred = torch.argmax(nn_logits, dim=1).item()
		nn_cam = get_cam(nn_C, nn_pred, cam_type=cam_type)

		new_dist, nn_coords = get_coords_nb_feature_in_nn(nn_C, nn_cam, query_feature)
		results.append([new_dist, nn_idx, nn_coords, nn_cam])

	results = sorted(results, key=lambda x: x[0], reverse=False) 
	
	xp_dist, closest_example_idx, xp_coord, xp_ab_cam = results[0]  

	xp_img = train_loader.dataset[NEIGHBORS[closest_example_idx]][0].view(1, 3, 224, 224)
	xp_logits, _, xp_C = netC(xp_img.to(DEVICE))

	x = xp_coord[1] * unit
	y = xp_coord[0] * unit
	h, w = unit, unit

	ccr_pixels = crop_ccr_cam_box(xp_img, x, y, h, w)

	_, nn_x, _ = netC(ccr_pixels.to(DEVICE))    


	if cam_type=='Random':
		ccr_pixels = torch.randn((1, 3, 224, 224))
		ccr_pixels = torch.clamp(ccr_pixels, -2, 2)
		_, query_x, _ = netC(ccr_pixels.to(DEVICE))



	# print(cam_type, 'Dist', torch.cdist(query_x, nn_x)  )

	return torch.cdist(query_x, nn_x)


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
		
		minr, minc, maxr, maxc = region.bbox

		# Get feature logit
		occlude_sp = keep_ccr_superpixel(image, segments, np.unique(segments)[idx], []) 
		PIL_image  = Image.fromarray(occlude_sp)
		torch_img  = transformNormalize(PIL_image).view(-1, 3, 224, 224)
		new_logits, _, _ = netC(torch_img.to(DEVICE))
		feature_logit = new_logits[0][QUERY_PRED].item()  
		
# 		plot_loader_img(torch_img)
# 		plt.show()
		
		# Get CCR x
		occlude_sp = crop_ccr_superpixel_box(occlude_sp, minr, minc, maxr, maxc, crop_image=True)
		PIL_image  = Image.fromarray(occlude_sp)
		PIL_image  = expand2square(PIL_image, background_color=0)
		torch_img  = transformNormalize(PIL_image).view(-1, 3, 224, 224)

		_, ccr_x, _ = netC(torch_img.to(DEVICE))
		
		results.append([feature_logit, idx+1, torch_img, ccr_x,  
								   None,  
								   region, SEGMENT_DIVISIONS, segments ])
				
	if keep_sp:
		results = sorted(results, key=lambda x: x[0], reverse=True) 

	if not keep_sp:
		results = sorted(results, key=lambda x: x[0], reverse=False) 
		
	# plot_loader_img(results[0][2])
	# plt.show()

	return results


def plot_loader_img(img, fam=False):
	img = deepcopy(img)
	unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	input_tensor = unorm(img)

	if fam:
		plt.imshow(input_tensor[0].permute(1,2,0), alpha=1)
	else:
		plt.imshow(input_tensor[0].permute(1,2,0))

	return input_tensor[0]


class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
		return tensor


def get_occlude_df_row_data():
	"""
	Get the row of data for all four methods
	meta has: 
	Logit drop
	Probability drop
	Probabaily of predicted class
	Class flip: boolean
	Num SP features gone
	% SP features gone
	Instance number in validation set
	
	Returns: df with all data
	"""
	
	sp   = get_sp_meta_data()
	pixels_occ = sp.PixelsOccluded.values
	cam  = get_cam_meta_data(pixels_occ, cam_type='CAM')
	fam  = get_cam_meta_data(pixels_occ, cam_type='FAM')
	rand = get_cam_meta_data(pixels_occ, cam_type='Random')
	
	all_data = pd.concat([sp, cam, fam, rand])
	return all_data


def blackout_ccr_superpixel(img, segments, segVal, occlude_pos=False):
	"""
	Choose to Keep only the sp area or not
	"""

	if occlude_pos:
		mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype="float32")
		mask += 1
		mask[segments == segVal] = 0
		mask = np.array([mask, mask, mask]).reshape(1, 3, IMG_SIZE, IMG_SIZE)
		mask = torch.tensor(mask).to(DEVICE)
		arr = img.to(DEVICE) * mask.to(DEVICE)
		
	else:
		mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype="float32")
		mask[segments == segVal] = 1
		mask = np.array([mask, mask, mask]).reshape(1, 3, IMG_SIZE, IMG_SIZE)
		mask = torch.tensor(mask).to(DEVICE)
		
		new_unmasked_area = QUERY_IMG.clone() * mask        
		arr = img.to(DEVICE) + new_unmasked_area.to(DEVICE)
	arr[arr == 0] = -2.1179
	return arr


def get_upsampled_cam_query(netC, cam_type='CAM', upsample=True):
	logits, x, C = netC(QUERY_IMG)
	pred = torch.argmax(logits, dim=1).item()
	
	if cam_type=='FAM':
		c = X_test_c[QUERY_IDX]
		nb_feature_idx = np.argmax(c)
		cam = C[0][nb_feature_idx].cpu().detach().numpy()  # really the FAM
		
	elif cam_type=='CAM':
		cam = get_cam(C, pred).cpu().detach().numpy()
		
	if cam_type=='Random':
		cam = np.random.rand(7,7)
		
	if upsample:
		cam = scipy.ndimage.zoom(cam, (32, 32), order=3) 
		return cam
	else:
		return cam


def get_cam_image_masked(img, cam, threshold, occlude_pos=True):
	"""
	Take in cam and num of pixels to change
	return a mask
	"""
	
	if occlude_pos:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape).to(DEVICE)
		mask += 1.

		for i in range(len(idx)):
			x, y = idx[i], idy[i]
			mask[x][y] = 0.

		img *= mask
		img[img==0] = -2.1179
	
	else:
		idx, idy = largest_indices(cam, threshold)
		mask = torch.zeros(cam.shape).to(DEVICE)

		for i in range(len(idx)):
			x, y = idx[i], idy[i]
			mask[x][y] = 1.

		img = QUERY_IMG.clone()
		img *= mask
		img[img==0] = -2.1179
	
	return img


def get_sp_meta_data():
	results = get_superpixel_ccrs()
	meta = pd.DataFrame()
	total_blacked_out = 0
	img = QUERY_IMG.clone()
	for i in range(len(results)):
		row = pd.DataFrame()

		#### Gradually introduce important parts
		sp_idx = results[i][1]
		segments = results[i][-1]
		img = blackout_ccr_superpixel(img, segments, sp_idx, occlude_pos=True)
		
# 		plot_loader_img(img.cpu())
# 		plt.show()

		# Change to black background
		temp = img.clone()
		temp[temp==0] = -2.1179

		logits, _, _ = netC(temp.to(DEVICE))
		logit = logits[0][QUERY_PRED].item()

		pred = torch.argmax(logits, dim=1).item()
		total_blacked_out += (segments == sp_idx).sum()
		prob = torch.softmax(logits, dim=1)[0][QUERY_PRED].item()

		row['Method'] = ['Superpixel']
		row['PixelsOccluded'] = int(round(total_blacked_out))
		row['LogitDrop'] = QUERY_LOGIT - logit
		row['Logit'] = logit
		row['QueryLogit'] = QUERY_LOGIT
		row['ClassSame'] = (QUERY_PRED == pred)
		row['InstanceIdx'] = QUERY_IDX
		row['QueryProb'] = QUERY_PROB.item()
		row['Prob'] = prob
		row['ProbDrop'] = (QUERY_PROB - prob).item()
		row['NumSpOccluded'] = i+1
		row['TotalSPs'] = len(results)

		meta = pd.concat([meta, row])

	nn_ccr_dist = get_sp_ccr_dist(QUERY_IMG, netC, results)
	meta['Dist_NN_CCR'] = nn_ccr_dist.item()

	return meta


def get_cam_meta_data(sp_occ_nums, cam_type='CAM'):
	meta = pd.DataFrame()
	img = QUERY_IMG.clone()
	total_blacked_out = 0
	for i in range(len(sp_occ_nums)):
		row = pd.DataFrame()
		
		#### For gradually including important parts
		pixel_cam = get_upsampled_cam_query(netC, cam_type=cam_type)
		masked_img = get_cam_image_masked(img, pixel_cam, sp_occ_nums[i], occlude_pos=True)
		logits, _, _ = netC(masked_img.to(DEVICE))
		feature_logit = logits[0][QUERY_PRED].item()
		logit = logits[0][QUERY_PRED].item()
		pred = torch.argmax(logits, dim=1).item()
		total_blacked_out += sp_occ_nums[i]
		prob = torch.softmax(logits, dim=1)[0][QUERY_PRED].item()
		
# 		plot_loader_img(img.cpu())
# 		plt.show()

		row['Method'] = [cam_type]
		row['PixelsOccluded'] = int(round(sp_occ_nums[i]))
		row['LogitDrop'] = QUERY_LOGIT - logit
		row['Logit'] = feature_logit
		row['QueryLogit'] = QUERY_LOGIT
		row['ClassSame'] = (QUERY_PRED == pred)
		row['InstanceIdx'] = QUERY_IDX
		row['QueryProb'] = QUERY_PROB.item()
		row['Prob'] = prob
		row['ProbDrop'] = (QUERY_PROB - prob).item()
		row['NumSpOccluded'] = i+1
		row['TotalSPs'] = len(sp_occ_nums)
		
		meta = pd.concat([meta, row])
		
	nn_ccr_dist = get_cam_ccr_dist(QUERY_IMG, netC, cam_type=cam_type)
	meta['Dist_NN_CCR'] = nn_ccr_dist.item()
		
	return meta


print("Start Testing Script... ======================")
for SEGMENT_DIVISIONS in [10, 20, 30, 40, 50]:

	print("SLIC Segments:", SEGMENT_DIVISIONS)

	torch.cuda.empty_cache()

	netC,  WEIGHTS  = collect_model(DEVICE)

	net_classifier = ClassifierCNN(netC)
	net_classifier = net_classifier.to(DEVICE).eval()

	expt_results = pd.DataFrame()
	start_time = time.time()

	for QUERY_IDX, data in enumerate(test_loader):

		if QUERY_IDX % 50 == 0:
			print(" ")
			print("Doing Query Number", QUERY_IDX, "in Testing Data Expt...")

		QUERY_IMG, QUERY_LABEL = data
		QUERY_IMG, QUERY_LABEL = QUERY_IMG.to(DEVICE), QUERY_LABEL.to(DEVICE)

		QUERY_PRED = test_preds[QUERY_IDX]
		QUERY_CONT = X_test_c[QUERY_IDX]
		logits, _, _ = netC(QUERY_IMG)
		QUERY_LOGIT = logits[0][QUERY_PRED].item()
		QUERY_PROB = torch.softmax(logits, dim=1)[0][QUERY_PRED]

		XP_IDXS = twin.kneighbors(X=[QUERY_CONT], n_neighbors=10, return_distance=False)
		NEIGHBORS = XP_IDXS[0]

		query_data = get_occlude_df_row_data()
		expt_results = pd.concat([expt_results, query_data])

		expt_results.to_csv('data/NewTestingResults/Testing_Segments_'  + str(SEGMENT_DIVISIONS) + '.csv')

		if QUERY_IDX == 500:
			break



	