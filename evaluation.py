import os
import torch

from torch import nn
import torch.optim as optim

from helpers.ARGS import ARGS
from helpers.helpers import plot_learning_curves
from learning.learner import validate_epoch
from learning.tunnel import Tunnel
from learning.utils import get_dataloader

def main():
    args = ARGS("DeepLabV3_Resnet50", "batch_17", len(Dataset.validClasses), labels_type="csv", batch_size=4, epochs=1)

    Dataset = Tunnel
    dataloaders = get_dataloader(Dataset, args)

    loss_fn = nn.CrossEntropyLoss()

    model, args = get_model(args)
    args.save_path = "save/DeepLabV3_Resnet50/2021-08-12-1628756459"

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    checkpoint = torch.load(os.path.join(args.save_path, "best_weights.pth.tar"))

    state = checkpoint["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    acc, loss, miou = validate_epoch(dataloaders['val'], model, loss_fn, 1, Dataset.classLabels, Dataset.validClasses, void=Dataset.voidClass, maskColors=Dataset.mask_colors, folder=args.save_path, args=args)

    print("\n\nEval on model {} after {} epochs".format(args.model, checkpoint["epoch"]))
    print("Pixel accuracy : {}".format(acc))
    print("Loss : {}".format(loss))
    print("Mean IoU: {}".format(miou))

if __name__ == "__main__":
    main()
