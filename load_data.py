import os
import shutil
import tempfile
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['validation'] = Subset(dataset, val_idx)
    return datasets

def LoadData(args):
    root_dir = 'Data/'

    transf = transforms.Compose([
        transforms.Resize([1664,1664]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    all_data = ImageFolder(root=root_dir, transform=transf)

    datasets = train_val_dataset(all_data)
    print(len(datasets['train']))
    print(len(datasets['validation']))
    dataloaders = {x:DataLoader(datasets[x],args.batchsize, shuffle=True, num_workers=4) for x in ['train','validation']}
    
    return dataloaders

# if __name__ == "__main__":
#     dataloaders = LoadData()
#     # x,y = next(iter(dataloaders['train']))
#     # print(x)

