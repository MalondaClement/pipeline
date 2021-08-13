#
#  augmentation/offline_data_aug.py
#
#  Created by Clément Malonda on 13/08/2021.

import os
from PIL import Image
import augly.image as imaugs

def main():
    dataset_path = "./tunnel"
    images_path = os.path.join(dataset_path, "images/train")
    labels_path = os.path.join(dataset_path, "labels/train")

    if not os.path.isdir(dataset_path):
        print("The dataset path {} is not correct".format(dataset_path))
        exit(1)
    if not os.path.isdir(labels_path):
        print("Before using data augmentation script, used create_labels.py script on your dataset")
        exit(1)

    



if __name__ == "__main__":
    main()
