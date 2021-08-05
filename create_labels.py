#
#  create_labels.py
#
#  Created by Clément Malonda on 29/07/2021.

import os
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

from datasets.tunnel import classToVal

# Dataset tree must be like :
#
# -── dataset
#     ├── images (all images are inside)
#     ├── labels (create in the fonction if it not exists)
#     └── targets (dir with all json file )
#

def createLabels(path, classToValDict, erase=True):
    new_images_counter = 0
    skip_images_counter = 0

    if not os.path.isdir(os.path.join(path, "targets")):
        print("Missing targets directory")
        exit(1)
    if not os.path.isdir(os.path.join(path, "images")):
        print("Missing images directory")
        exit(1)
    if not os.path.isdir(os.path.join(path, "labels")):
        print("create rep {} to save results".format(os.path.join(path, "labels")))
        os.makedirs(os.path.join(path, "labels"))

    for json_file in os.listdir(os.path.join(path, "targets")):
        if json_file[-5:] != ".json":
            continue # skip non .json file
        file = open(os.path.join(path, "targets", json_file))
        targets_data = json.load(file)
        for target_data in targets_data:
            if target_data["name"] not in os.listdir(os.path.join(path, "images")):
                continue # skip image if not exist in images directory
            if target_data["name"] in os.listdir(os.path.join(path, "labels")) and not erase:
                skip_images_counter+=1
                continue # skip image if
            else :
                new_images_counter+=1
            image = Image.open(os.path.join(path, "images", target_data["name"])).convert("RGB")
            target = Image.new("L", (image.size[0], image.size[1]))
            draw = ImageDraw.Draw(target)
            for label in target_data["labels"]["polygonLabels"]:
                x = np.array(label["allX"], dtype="int32")
                y = np.array(label["allY"], dtype="int32")
                points = list(zip(x, y))
                draw.polygon(points, fill=classToValDict[label["labelValue"]])
            target.save(os.path.join(path, "labels", target_data["name"]))
        file.close()
        print("New labels images : {} \nLabels images skip : {}".format(new_images_counter, skip_images_counter))


def main():
    createLabels("tunnel", classToVal, False)

if __name__ == "__main__":
    main()
