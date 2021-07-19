#
#  trainUNet.py
#
#  Created by Clément Malonda on 15/07/2021.

import os
import torch

from torch import nn
import torch.optim as optim

from models.UNet import UNet

from helpers.ARGS import ARGS
from helpers.helpers import plot_learning_curves
from learning.learner import train_epoch, validate_epoch
from learning.cityscapes import Cityscapes
from learning.utils import get_dataloader

def main():
    # Set up execution arguments
    args = ARGS()
    args.epochs = 200
    args.batch_size = 16
    args.model = "UNet"
    args.save_path = "UNet_save"
    args.is_pytorch_model = False
    args.dataset_path = "cityscapes"

    # Get cityscapes dataset and dataloaders
    Dataset = Cityscapes
    dataloaders = get_dataloader(Dataset, args)


    # Create all directories for res
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path + '/images')
    if not os.path.isdir(args.save_path + '/results_color_val'):
        os.makedirs(args.save_path + '/results_color_val')
        os.makedirs(args.save_path + '/results_color_test')

    # Set model
    model = UNet(n_class=30)

    # Check if cuda is available to push model on GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Get loss and optimizer functions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = 0.0
    metrics = {'train_loss' : [],
                'train_acc' : [],
                'val_acc' : [],
                'val_loss' : [],
                'miou' : []}

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(dataloaders["train"], model, loss_fn, optimizer, scheduler, epoch, args=args)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,train_loss,train_acc))

        val_acc, val_loss, miou = validate_epoch(dataloaders['val'], model, loss_fn, epoch, Dataset.classLabels, Dataset.validClasses, void=Dataset.voidClass, maskColors=Dataset.mask_colors, folder=args.save_path, args=args)

        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(val_loss)
        metrics['miou'].append(miou)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'metrics': metrics,
            }, args.save_path + '/checkpoint.pth.tar')

        if miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou))
            best_miou = miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, args.save_path + '/best_weights.pth.tar')

    # Create learning curve with loss, miou and accuracy for each epoch
    plot_learning_curves(metrics, args)

if __name__ == "__main__":
    main()
