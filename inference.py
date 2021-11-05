#
#  train.py
#
#  Created by Clément Malonda on 02/08/2021.

import os
import time
import torch
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from PIL import Image

from helpers.ARGS import ARGS
from helpers.helpers import vislbl
from datasets.tunnel import Tunnel
from models.utils import get_model

def main():
    # Get tunnel dataset
    Dataset = Tunnel

    # Set up execution arguments
    args = ARGS("DeepLabV3_Resnet50", "tunnel", len(Dataset.validClasses), labels_type="csv", batch_size=2, epochs=2)

    model, args = get_model(args)
    args.save_path = "../"

    checkpoint = torch.load(os.path.join(args.save_path, "best_weights.pth.tar"), map_location=torch.device('cpu'))

    print("Model {} à l'epoch {} ".format(args.model, checkpoint["epoch"]))

    if not os.path.isdir(os.path.join(args.save_path, "inference")) :
        os.makedirs(os.path.join(args.save_path, "inference"))

    state = checkpoint["model_state_dict"]
    new_state = {}
    for key in state:
        new_state[key[7:]] = state[key]

    model.load_state_dict(new_state)
    model.eval()

    times = list()

    start = time.time()
    for i, file in enumerate(os.listdir("/Users/ClementMalonda/Desktop/img_inf")):
        if file[-4:] != ".png":
            continue
        start = time.time()

        img = Image.open(os.path.join("/Users/ClementMalonda/Desktop/img_inf",file))
        img = np.array(img)
        img = img[:,:,:3]
        img = img/255

        input = ToTensor()(img)
        input = input.unsqueeze(0)
        input = input.float()
        with torch.no_grad():
            output = model(input)
        if args.is_pytorch_model:
            output = output["out"]
        preds = torch.argmax(output, 1)
        pred = preds[0,:,:].cpu()
        pred_color = vislbl(pred, Dataset.mask_colors)
        pred_color = Image.fromarray(pred_color.astype("uint8"))
        end = time.time()
        times.append(int(end - start))
        print("Temps d'inférence sur l'image {}: {} min {} s".format(file, int((end - start)/60), int((end - start)%60)))

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        fig.set_size_inches(18.5, 10.5)
        ax0.imshow(img)
        ax0.set_title("Image d'origine")
        ax1.imshow(pred_color)
        ax1.set_title("Prédiction")
        ax2.imshow(img)
        ax2.imshow(pred_color, alpha=0.6)
        ax2.set_title("Superposition de l'image avec la prédiction")
        fig.savefig(os.path.join(args.save_path, "inference", file))

    mean = int(np.mean(times))
    print("Temps d'inférence moyen sur les images : {} min  {} s".format(int(mean/60), mean%60))
    # plt.show()

if __name__ == "__main__":
    main()
