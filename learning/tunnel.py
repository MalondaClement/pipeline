#
#  learning/tunnel.py
#
#  Created by Clément Malonda on 28/07/2021.

import os
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

classToVal = {"BAU": 1, "Parking" : 2, "SW": 3, "Road": 4}

class Tunnel():
    id2trainid = np.array([0,1,2,3,4], dtype="uint8")

    mask_colors = np.array([[0,0,0], [81,0,81], [230,150,140], [244,35,232], [128,64,128]])

    validClasses = [0,1,2,3,4]

    classLabels = ["Background", "BAU", "Parking", "SW", "Road"]

    def __init__(self, root, split="train", transform=None, target_transform=None, transforms=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.images_dir = os.path.join(self.root, "images")
        self.target_dir = os.path.join(self.root, "targets")
        self.split = split
        self.images = list()
        self.targets = dict()

        assert split in ["train","val","test"], "Unknown value {} for argument split.".format(split)

        for e in os.listdir(self.target_dir):
            if e[-5:] == ".json":
                file = open(os.path.join(self.target_dir, e))
                targets_data = json.load(file)
                for target_data in targets_data:
                    if target_data["name"] in os.listdir(self.images_dir):
                        self.images.append(os.path.join(self.images_dir, target_data["name"]))
                        poly = list()
                        for label in target_data["labels"]["polygonLabels"]:
                            x = np.array(label["allX"], dtype="int32")
                            y = np.array(label["allY"], dtype="int32")
                            points = list(zip(x, y))
                            poly.append((points, label["labelValue"]))
                        self.targets.update({os.path.join(self.images_dir, target_data["name"]) :  poly})

                file.close()





    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert("RGB")
        # target = np.zeros((image.size[1], image.size[0]))
        target = Image.new("RGB", (image.size[0], image.size[1]))
        draw = ImageDraw.Draw(target)
        for e in self.targets[filepath]:
            draw.polygon(e[0], fill=classToVal[e[1]])
        image = np.array(image)
        image = image.transpose(2, 0, 1)
        target = np.array(target)[:, :, 0]
        return image, target, filepath

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = Tunnel
    t = dataset("/Users/ClementMalonda/Desktop/tunnel")
    image, target, filepath = t[0]
    plt.imshow(target)
    plt.show()
