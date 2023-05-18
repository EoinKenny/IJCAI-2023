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

from functions import *


NAME = 'NORMAL'
NGPU = 1
DEVICE = 'cuda:1'
BATCH_SIZE = 12
NUM_WORKERS = 4
IMG_SIZE = 224
UNIT = 32.
ALPHA = None
NUM_EPOCHS = 1


def get_num_ccrs(cam):
    results = list()
    max_saliency = cam.max()
    threshold = max_saliency / ALPHA
    return sum(cam.flatten() >= threshold)


train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(shuffle=True,
                                                                         training_transform=True,
                                                                         b_size=BATCH_SIZE)



netC, _  = collect_model(DEVICE)
netC2, WEIGHTS = collect_model(DEVICE)


def get_cam(C, pred):
    test_weights = WEIGHTS[pred.item()]
    C_conts = test_weights.reshape(test_weights.shape[0],1,1) * C
    return C_conts.sum(axis=0)


def blackout_img(img, blackout_segs, rand_blackout=False):

    if rand_blackout:
        # Lose n Random
        rand_blackout_segs = list()
        while len(rand_blackout_segs) < len(blackout_segs):
            i, j = random.randint(0, 6), random.randint(0, 6)
            if [i, j] not in rand_blackout_segs:
                rand_blackout_segs.append([i, j])

        for xxx, coords in enumerate(rand_blackout_segs):
            i, j = coords
            img[:, int(i*UNIT): int(i*UNIT+UNIT), int(j*UNIT): int(j*UNIT+UNIT)] = -2.11

    else:
        # Lose n most important
        for idx_current, coords in enumerate(blackout_segs):
            i, j = coords
            img[:, int(i*UNIT): int(i*UNIT+UNIT), int(j*UNIT): int(j*UNIT+UNIT)] = -2.11

    return img


def evaluate_validation(test_loader, netC, test_dataset, DEVICE):

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

#     print( "\n Validation Accuracy Top 1: " + str(top1_correct.item() / test_loader.dataset.data.target.shape[0] ) )
#     print( "\n Validation Accuracy Top 5: " + str(top5_correct.item() / test_loader.dataset.data.target.shape[0] ) )
    
    return top1_correct.item() / test_loader.dataset.data.target.shape[0]


def largest_indices(ary, n):
    """Returns the n largest indices from a 2d numpy array."""
    
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def get_fam(C, pred):
    """
    Get FAM in C
    return: 2D array of FAM
    """
    
    gap = torch.nn.AvgPool2d(7)
    x = gap(C).view(512)
    c = x * WEIGHTS[pred]
    fam_idx = torch.argmax(c)
    fam = C[fam_idx]
    
    return fam


def get_img_nb_coords(C, pred, use_fam=False):
    
#     if BLACKOUT_NUM == 0:
#         return []
    
    cam = get_cam(C, pred).cpu().detach().numpy()
    fam = get_fam(C, pred).cpu().detach().numpy()
    
    
    # Num CCRs based on Alpha -- Or use BLACKOUT_NUM for a constrained amount
    num_ccrs = get_ccr_num(cam)
    
    ccr_indices = largest_indices(cam, 49)
    fam_indices = largest_indices(fam, 49)
        
    
    if use_fam:
        results = list()
        for c in zip(fam_indices[0][num_ccrs:], fam_indices[1][num_ccrs:]):
            results.append([c[0], c[1]])
        return results
    
    else:
        results = list()
        for c in zip(ccr_indices[0][num_ccrs:], ccr_indices[1][num_ccrs:]):
            results.append([c[0], c[1]])
        return results


def get_ccr_num(cam):
    alpha = max(cam.flatten()) / ALPHA
    
    total_ccrs = 0
    
    for i in range(7):
        for j in range(7):
            if cam[i][j] >= alpha:
                total_ccrs += 1
                
    return total_ccrs


def retrain(use_fam=False, rand_blackout=False):
    netC, _ = collect_model(DEVICE)
    netC2, WEIGHTS = collect_model(DEVICE)
    weights = netC2.linear.weight
    cce_loss = torch.nn.CrossEntropyLoss()
    optimizerC = torch.optim.SGD(netC.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

    netC = netC.train()
    netC2 = netC2.eval()

    acc = evaluate_validation(test_loader, netC.eval(), test_dataset, DEVICE)
    
    accs = [acc]
    iterations = [0]
    epochs = [0]
    current_iter = 0
    avg_size = list()
    
    for epoch in range(NUM_EPOCHS):
    

        for i, data in enumerate(train_loader):
            
            current_iter += 1

            netC.zero_grad()
            netC.train()

            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                logits, _, Cs = netC2(imgs)
                preds = torch.argmax(logits, dim=1)
                for k in range(len(imgs)):
                    b_coords = get_img_nb_coords(Cs[k], preds[k], use_fam=use_fam)
                    imgs[k] = blackout_img(imgs[k], b_coords, rand_blackout=rand_blackout)
                    avg_size.append(len(b_coords))

            logits, x, C = netC(imgs)
            preds = torch.argmax(logits, dim=1)
            loss = cce_loss(logits, labels)

            loss.backward()
            optimizerC.step()
            
            print(100 * ( (sum(avg_size) / len(avg_size))  / 49  ))

            if current_iter % 50 == 0:
                acc = evaluate_validation(test_loader, netC.eval(), test_dataset, DEVICE)
                accs.append(acc)
                iterations.append(current_iter)
                epochs.append(epoch)
                print(current_iter, acc, epoch)
            
    return accs, iterations, epochs, 100 * ( (sum(avg_size) / len(avg_size))  / 49  )


for ALPHA in [1, 10.5, 1.1, 1.25, 1.5, 2, 3]:

    accs, iterations, epochs, avg_size = retrain(use_fam=False, rand_blackout=False)

    df1 = pd.DataFrame()
    df1['Accuracy'] = accs
    df1['Iterations'] = iterations
    df1['Technique'] = 'CAM'
    df1['Epochs'] = epochs
    df1['Avg Size'] = avg_size

    accs, iterations, epochs, avg_size = retrain(use_fam=True, rand_blackout=False)

    df2 = pd.DataFrame()
    df2['Accuracy'] = accs
    df2['Iterations'] = iterations
    df2['Technique'] = 'FAM'
    df2['Epochs'] = epochs
    df1['Avg Size'] = avg_size

    accs, iterations, epochs, avg_size = retrain(use_fam=True, rand_blackout=True)

    df3 = pd.DataFrame()
    df3['Accuracy'] = accs
    df3['Iterations'] = iterations
    df3['Technique'] = 'Random'
    df3['Epochs'] = epochs
    df1['Avg Size'] = avg_size

    import seaborn as sns

    df = pd.concat([df1, df2, df3])

    df = df.replace({'CAM': 'CAM CCRs', 'FAM': "FAM CCRs", "Random": "Random CCRs"})

    df.Accuracy *= 100

    df.to_csv('alpha'  + str(ALPHA) + '_cub.csv')



