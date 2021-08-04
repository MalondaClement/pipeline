#
#  learning/tunnel.py
#
#  Created by Clément Malonda on 28/07/2021.

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

classToVal = {"BAU": 1, "Parking" : 2, "SW": 3, "Road": 4}

class Tunnel():

    voidClass = 5

    id2trainid = np.array([0,1,2,3,4], dtype="uint8")

    mask_colors = np.array([[0,0,0], [81,0,81], [230,150,140], [244,35,232], [128,64,128]])

    validClasses = [0,1,2,3,4]

    classLabels = ["Background", "BAU", "Parking", "SW", "Road"]

    def __init__(self, root, split="train", labels_type="csv", transform=None, target_transform=None, transforms=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.images_dir = os.path.join(self.root, "images")
        self.split = split
        self.images = list()
        self.targets = dict()
        self.labels_type = labels_type

        assert split in ["train","val","test"], "Unknown value {} for argument split.".format(split)
        if self.labels_type == "json":
            self.target_dir = os.path.join(self.root, "jsons")
            self.__read_csv()
        elif self.labels_type == "csv":
            self.target_dir = os.path.join(self.root, "csvs")
            self.__read_csv()
        else :
            print("args.labels_type can only be json or csv")
            exit(1)


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

    def __read_json(self):
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

    def __read_csv(self):
        datas = pd.read_csv(os.path.join(self.target_dir, "all.csv"))
        exp1 = re.compile("/A[0123456789]")
        exp2 = re.compile(".png")
        image_name = ""
        for data in datas.iterrows() :
            p1 = exp1.search(data[1][2])
            p2 = exp2.search(data[1][2])
            if p1 != None and p2 != None :
                i, _ = p1.span()
                _, j = p2.span()
                tmp = data[1][2][i+1:j]
                if tmp != image_name :
                    if image_name != "":
                        self.targets.update({os.path.join(self.images_dir, target_data["name"]) :  poly})
                    image_name = tmp
                    poly = list()
                if type(data[1][5]) == type(" ") :
                    label = data[1][4]
                    allX = "["+data[1][5]+"]"
                    allX = json.loads(allX.replace(";", ","))
                    allY = "["+data[1][6]+"]"
                    allY = json.loads(allY.replace(";", ","))
                    if len(allX) >= 2 :
                        points = list(zip(allX, allY))
                        poly.append((points, label))


if __name__ == "__main__":
    dataset = Tunnel
    t = dataset("/Users/ClementMalonda/Desktop/tunnel")
    image, target, filepath = t[0]
    plt.imshow(target)
    plt.show()
