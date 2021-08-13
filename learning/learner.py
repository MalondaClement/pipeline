#
#  learning/learner.py
#
#  Created by Clément Malonda on 15/07/2021.

from helpers.helpers import AverageMeter, ProgressMeter, iouCalc, visim, vislbl
from learning.utils import rand_bbox, copyblob
import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


# Function used to train the model on one epoch
def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, validClasses, void=-1, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    # input resolution
    if args.crop_size is not None:
        res = args.crop_size[0]*args.crop_size[1]
    else:
        res = args.train_size[0]*args.train_size[1]

    # Set model in training mode
    model.train()

    end = time.time()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels, _) in enumerate(dataloader):
            data_time.update(time.time()-end)

            #test
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
            images = inputs.numpy()
            image = images[0, :, :, :]
            image = image.transpose(1, 2, 0)
            ax0.imshow(image)
            ax0.set_title("Image d'origine")
            print(type(inputs))
            print(inputs.shape)
            #end test

            if args.copyblob:
                for i in range(inputs.size()[0]):
                    rand_idx = np.random.randint(inputs.size()[0])
                    # wall(3) --> sidewalk(1)
                    copyblob(validClasses, src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=3, dst_class=1)
                    # fence(4) --> sidewalk(1)
                    copyblob(validClasses, src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=4, dst_class=1)
                    # bus(15) --> road(0)
                    copyblob(validClasses, src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=15, dst_class=0)
                    # train(16) --> road(0)
                    copyblob(validClasses, src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=16, dst_class=0)

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            if args.cutmix:
                # generate mixed sample
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]

            # forward pass
            outputs = model(inputs)
            if args.is_pytorch_model :
                outputs = outputs['out'] #FIXME for DeepLab V3
            preds = torch.argmax(outputs, 1)

            #test
            print(type(preds))
            print(preds.shape)
            pred = preds[0, :, :].cpu()
            ax1.imshow(pred)
            ax1.set_title("Prédiction")
            ax2.imshow(pred)
            ax2.imshow(pred, alpha=0.6)
            ax2.set_title("Superposition de l'image avec la prédiction")
            if not os.isdir(os.path.join(args.save_path, "inference")):
                os.makedirs(os.path.join(args.save_path, "inference"))
            fig.savefig(os.path.join(args.save_path, "inference", str(epoch)+".png"))
            # end test

            # cross-entropy loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)

            # output training info
            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step(loss_running.avg)

    return loss_running.avg, acc_running.avg


# Function used to validate the model
def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None, folder='baseline_run', args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.4e')
    iou = iouCalc(classLabels, validClasses, voidClass = void)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Test, epoch: [{}]".format(epoch))

    # input resolution
    res = args.test_size[0]*args.test_size[1]

    # Set model in evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

            # forward
            outputs = model(inputs)
            if args.is_pytorch_model :
                outputs = outputs['out'] #FIXME
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress info
            progress.display(epoch_step)

        miou = iou.outputScores()
        print('Accuracy      : {:5.3f}'.format(acc_running.avg))
        print('---------------------')

    return acc_running.avg, loss_running.avg, miou
