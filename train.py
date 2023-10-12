import utils
import logging
import os
import sys
from torch.utils.data import DataLoader
from glob import glob

import torch
from PIL import Image
import argparse
import monai
from monai.data import PersistentDataset, list_data_collate, SmartCacheDataset, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai import transforms as mt
from monai.visualize import plot_2d_or_3d_image
import random
import ufcn
import load_FFSData
import load_FFSDataTest
import processing

pjoin = os.path.join

def get_transforms():
    train_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.EnsureChannelFirstd(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img',"seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
        ]
    )

    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.EnsureChannelFirstd(keys=["img", "seg"]),
            mt.ScaleIntensityD(keys=['img',"seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
        ]
    )
    return train_trans, val_trans

def main(args):

    dataset = load_FFSData.CustomDataset()
    data_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    print("number batch of train",data_loader.__len__())
    testDataset = load_FFSDataTest.CustomDataset()
    
    testDataload = DataLoader(testDataset, batch_size=args.batchsize, shuffle=True)
    print("number batch of test",testDataload.__len__())
    print("data load done")


    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = mt.Compose([
        mt.Activations(sigmoid=True)
    ])

    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ufcn.UFCN(activation = args.activation, threshold = args.thresh)

    #model.apply(utils.init_weights)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    print("model loaded")
    loss_function = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)
    bce = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    outputPreq = 0
 
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for inputs1, inputs2, labels in data_loader:
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            step += 1
            data_range = inputs1.max().unsqueeze(0)
            outputs, _, labelLoss  = model(inputs1, inputs2)
       
            l1 = loss_function(outputs, inputs1, data_range)

            term = 1
            layerLoss = 0
            for i in labelLoss:
                layerLoss = layerLoss + term * bce(i.to(dtype = float), labels.to(dtype = float))
                term -= 0.4

            l2 = layerLoss / len(labelLoss)

            loss =  0.9 * l1 + 0.1 * l2
            #print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #epoch_len = len(train_ds) // data_loader.batch_size
            #print(f"{step}, train_loss: {loss.item():.4f}")
           
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_inter == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for inputs1, inputs2, label, seg in testDataload:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    label = label.to(device)
                    seg = seg.to(device)

                    val_outputs, val_seg, labelLoss = model(inputs1, inputs2)

                    if label == 1:
                        value = dice_metric(y_pred=processing(val_seg), y=processing(seg))
                        metric_count += len(value)
                        
                        metric_sum += value.item() * len(value)
                        
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), pjoin('checkpoints', f'{args.arch}_{args.ext}.pth'))
                    print("saved new best metric model")
                    print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                     )

            scheduler.step(metric)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--data", default='gd', type=str)
    parser.add_argument("--arch", default='ufcn', type=str)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--fast", default=False, type=bool)
    parser.add_argument("--dataDic", default='./train_data')
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--ext", default='unet', type=str)
    parser.add_argument("--pref", default=20, type=int)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--thresh", default=0.02, type=float)
    args = parser.parse_args()
    print(args)
    
    main(args)