#
#  create_labels.py
#
#  Created by Clément Malonda on 29/07/2021.

import os
import re
import json
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

from datasets.tunnel import classToVal

# Dataset tree must be like :
#
# -── dataset
#     ├── images (all images are inside)
#     │   ├── train
#     │   └── test
#     ├── labels (create in the fonction if it not exists)
#     │   ├── train (create in the fonction if it not exists)
#     │   └── test (create in the fonction if it not exists)
#     ├── jsons (create in the fonction if it not exists)
#     └── csvs (dir with all json file )
#

def read_csv(path, split="train"):
    datas = pd.read_csv(os.path.join(path, "csvs", "all.csv"))
    exp1 = re.compile("/A[0123456789]")
    exp2 = re.compile(".png")
    image_name = ""
    images = list()
    labels = dict()

    if not os.path.isdir(os.path.join(path, "labels", split)):
        os.makedirs(os.path.join(path, "labels", split))

    for data in datas.iterrows() :
        p1 = exp1.search(data[1][2])
        p2 = exp2.search(data[1][2])
        if p1 != None and p2 != None :
            i, _ = p1.span()
            _, j = p2.span()
            tmp = data[1][2][i+1:j]
            if tmp not in os.listdir(os.path.join(path, "images", split)):
                continue
            if tmp != image_name :
                if image_name != "":
                    images.append(os.path.join(path, "images", split, image_name))
                    labels.update({os.path.join(path, "images", split, image_name) :  poly})
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
    return images, labels

def read_json(path, split="train"):
    pass

def from_poly_to_labels(images, poly):
    for filepath in images:
        print(filepath)
        image = Image.open(filepath).convert("RGB")
        label = Image.new("P", (image.size[0], image.size[1]))
        draw = ImageDraw.Draw(label)
        for e in poly[filepath]:
            draw.polygon(e[0], fill=classToVal[e[1]])



def create_labels(path, classToValDict, labels_type="csv", erase=True):
    new_images_counter = 0
    skip_images_counter = 0

    if labels_type not in ["csv", "json"]:
        print("Labels file type must be csv or json")
        exit(1)
    if not os.path.isdir(os.path.join(path, "images/train")):
        print("Missing images/train directory")
        exit(1)
    if not os.path.isdir(os.path.join(path, "images/test")):
        print("Missing images/test directory")
        exit(1)
    if not os.path.isdir(os.path.join(path, "labels")):
        print("Create {} directory to save results".format(os.path.join(path, "labels")))
        os.makedirs(os.path.join(path, "labels"))

    if labels_type == "csv":
        # images, labels = read_csv(path, split="train")
        # from_poly_to_labels(images, labels)
        images, labels = read_csv(path, split="test")
        from_poly_to_labels(images, labels)

    elif labels_type == "json":
        pass

    # print("New labels images : {} \nLabels images skip : {}".format(new_images_counter, skip_images_counter))

def main():
    create_labels("/Users/ClementMalonda/Desktop/batch_17_test_create_label", classToVal, labels_type="csv", erase=True)

if __name__ == "__main__":
    main()
