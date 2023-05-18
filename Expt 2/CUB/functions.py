import pandas as pd
import numpy as np
import pickle
import time
import scipy
import matplotlib.pyplot as plt
import torchvision.models as models
import os
import pandas as pd
import torch 
import torchvision.transforms as transforms

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader


IMG_SIZE = 224
NUM_WORKERS = 4


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])

        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_dataloaders(shuffle=True, training_transform=True, b_size=1):

    root = 'data/'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
#         transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.ToTensor(),
        normalize
        ])

    test_transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                        transforms.CenterCrop(IMG_SIZE),
                                        transforms.ToTensor(),
                                        normalize
                                        ])

    if not training_transform:
        train_transform = test_transform

    train_dataset = Cub2011(root, train=True,  transform=train_transform, loader=default_loader, download=False)
    test_dataset  = Cub2011(root, train=False, transform=test_transform,  loader=default_loader, download=False)



    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=b_size,
                              shuffle=shuffle,
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=b_size,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    return train_loader, test_loader, train_dataset, test_dataset
    
    
class NetC(torch.nn.Module):
	
	def __init__(self, net):
		super(NetC, self).__init__()
		self.net = net
		self.main = torch.nn.Sequential(*list(self.net.children()))[:-2]
		self.avgpool = torch.nn.AvgPool2d(7)
		self.linear = torch.nn.Linear(512, 200) 

	def forward(self, imgs):
		C = self.main(imgs)      # run resnet part of CNN
		x = self.avgpool(C)   # Focus only on one part of these new feature maps
		x = x.view(x.shape[0], x.shape[1])
		logits = self.linear(x)
		return logits, x, C

    
class ClassifierCNN(torch.nn.Module):
	
	def __init__(self, net):
		super(ClassifierCNN, self).__init__()
		self.net = net
		self.main = torch.nn.Sequential(*list(self.net.children()))[:-2]
		self.avgpool = torch.nn.AvgPool2d(7)
		self.linear = torch.nn.Linear(512, 200) 

	def forward(self, C):
		x = self.avgpool(C)   # Focus only on one part of these new feature maps
		x = x.view(x.shape[0], x.shape[1])
		logits = self.linear(x)
		return logits

    
    
def collect_model(DEVICE):
    resnet = models.resnet34(pretrained=True).train()
    netC = NetC(resnet)
    netC.load_state_dict(torch.load('weights/cnnNORMAL.pth'))
    netC.to(DEVICE)
    classifier_cnn = ClassifierCNN(netC)
    return netC.eval(), netC.linear.weight
    
    
def evaluate_test(test_loader, netC, test_dataset, DEVICE):
    total_correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits, _, _ = netC(imgs)
            preds = torch.argmax(logits, dim=1)

            total_correct += sum(preds == labels).item()

    print("Accuracy:", total_correct / test_dataset.data.shape[0] )
    
    
    
def evaluate_train(train_loader, netC, train_dataset, DEVICE):
    total_correct = 0

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits, _, _ = netC(imgs)
            preds = torch.argmax(logits, dim=1)

            total_correct += sum(preds == labels).item()

    print("Accuracy:", total_correct / train_dataset.data.shape[0] )