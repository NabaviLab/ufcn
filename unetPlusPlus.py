 
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
import process

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


    images = sorted(glob(os.path.join('data/train/img/', "*.png")))
    segs = sorted(glob(os.path.join("data/train/seg/", "*.png")))

    vaimages = sorted(glob(os.path.join('data/test/img/', "*.png")))
    vasegs = sorted(glob(os.path.join("data/test/seg/", "*.png")))

    train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(vaimages, vasegs)]
    random.shuffle(train_files)
    random.shuffle(val_files)

    train_trans, val_trans = get_transforms()
    train_ds = PersistentDataset(data=train_files, transform=train_trans,cache_dir='./train_cache',pickle_protocol=4)
    val_ds = PersistentDataset(data=val_files, transform=val_trans,cache_dir='./val_cache',pickle_protocol=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1,
                              pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size,
                            num_workers=1)  # , pin_memory=torch.cuda.is_available())
    # define transforms for image and segmentation
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = mt.Compose([
        mt.Activations(sigmoid=True),
        mt.AsDiscrete(threshold_values=True),
    ])

    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = monai.networks.nets.UNet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=(64, 128, 256, 512, 1024),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # )

    model = monai.networks.nets.BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 64)
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    print("model loaded")
    loss_function = monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    outputPreq = 0
 
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs[0].shape)
            loss = loss_function(outputs[0], labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
           
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
                for val_data in val_loader:
                    val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
                    val_outputs = model(val_images)

                    val_outputs = post_trans(val_outputs[0])

                    # if count == 30:
                    #     cpu_pred = val_outputs.cpu()
                    #     result = cpu_pred.data.numpy()
                    #     np.save(result, )
                    
                    value = dice_metric(y_pred=process(val_outputs), y=process(val_labels))
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)

                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), pjoin('checkpoints', f'{args.arch}_best300.pth'))
                    print("saved new best metric model")
                    print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                     )


                fldr = "plot/ultra_" + args.ext
                try:
                    os.makedirs(fldr, exist_ok=True)
                except TypeError:
                    raise Exception("Direction not create!")
            scheduler.step(metric)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--data", default='gd', type=str)
    parser.add_argument("--arch", default='unetPlusPlus', type=str)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--fast", default=False, type=bool)
    parser.add_argument("--dataDic", default='./train_data')
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--ext", default='unet', type=str)
    parser.add_argument("--pref", default=20, type=int)
    args = parser.parse_args()
    print(args)
    
    main(args)
