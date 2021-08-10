#
#  train.py
#
#  Created by Clément Malonda on 30/07/2021.

import os
import torch

from torch import nn
import torch.optim as optim

from models.utils import get_model
from helpers.ARGS import ARGS
from helpers.helpers import plot_learning_curves
from learning.learner import train_epoch, validate_epoch
from learning.utils import get_dataloader
# from datasets.tunnel import Tunnel
from datasets.minicity import MiniCity

def main():
    # Get tunnel dataset
    Dataset = MiniCity

    # Set up execution arguments
    args = ARGS("DeepLabV3_Resnet50", "microcity", len(Dataset.validClasses), labels_type="csv", batch_size=2, epochs=40)

    # Get model
    model, args = get_model(args)

    # Get tunnel dataloaders
    dataloaders = get_dataloader(Dataset, args)

    # Create save directory
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # Check if cuda is available to push model on GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # TODO: add loss optim and sheduler in get_model
    # Get loss and optimizer functions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = 0.0
    metrics = {'train_loss' : [], 'train_acc' : [], 'val_acc' : [], 'val_loss' : [], 'miou' : []}

    # Training loop
    for epoch in range(args.epochs):
        # Train the model for an epoch
        train_loss, train_acc = train_epoch(dataloaders["train"], model, loss_fn, optimizer, scheduler, epoch, Dataset.validClasses, args=args)

        # Save trains metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,train_loss,train_acc))

        # Validate epoch
        val_acc, val_loss, miou = validate_epoch(dataloaders['val'], model, loss_fn, epoch, Dataset.classLabels, Dataset.validClasses, void=Dataset.voidClass, maskColors=Dataset.mask_colors, folder=args.save_path, args=args)

        # Save val metrics
        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(val_loss)
        metrics['miou'].append(miou)

        # Save checkpoint on
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_miou': best_miou, 'metrics': metrics}, args.save_path + '/checkpoint.pth.tar')

        if miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou))
            best_miou = miou
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, args.save_path + '/best_weights.pth.tar')

    # Create learning curve with loss, miou and accuracy for each epoch
    plot_learning_curves(metrics, args)


if __name__ == "__main__":
    main()
