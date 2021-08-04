#
#  models/utils.py
#
#  Created by Clément Malonda on 30/07/2021.

import os
import time
from datetime import date

def get_model(args):
    if args.model == "DeepLabV3_Resnet50":
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(pretrained=False, num_classes = args.num_classes)
        args.is_pytorch_model = True

    elif args.model == "DeepLabV3_Resnet101":
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(pretrained=False, num_classes = args.num_classes)
        args.is_pytorch_model = True

    elif args.model == "FCN_Resnet50":
        from torchvision.models.segmentation import fcn_resnet50
        model = fcn_resnet50(pretrained=False, num_classes = args.num_classes)
        args.is_pytorch_model = True

    elif args.model == "FCN_Resnet101":
        from torchvision.models.segmentation import fcn_resnet101
        model = fcn_resnet101(pretrained=False, num_classes = args.num_classes)
        args.is_pytorch_model = True

    elif args.model == "UNet":
        from models.UNet import UNet
        model = UNet(num_classes=args.num_classes)
        args.is_pytorch_model = False

    elif args.model == "DenseASPP121":
        from models.DenseASPP import DenseASPP
        from models.configs.DenseASPP121 import Model_CFG
        model = DenseASPP(Model_CFG, num_classes=args.num_classes)
        args.is_pytorch_model = False

    elif args.model == "DenseASPP161":
        from models.DenseASPP import DenseASPP
        from models.configs.DenseASPP161 import Model_CFG
        model = DenseASPP(Model_CFG, num_classes=args.num_classes)
        args.is_pytorch_model = False

    elif args.model == "DenseASPP169":
        from models.DenseASPP import DenseASPP
        from models.configs.DenseASPP169 import Model_CFG
        model = DenseASPP(Model_CFG, num_classes=args.num_classes)
        args.is_pytorch_model = False

    elif args.model == "DenseASPP201":
        from models.DenseASPP import DenseASPP
        from models.configs.DenseASPP201 import Model_CFG
        model = DenseASPP(Model_CFG, num_classes=args.num_classes)
        args.is_pytorch_model = False

    elif args.model == "MobileNetDenseASPP":
        from models.DenseASPP import DenseASPP
        from models.configs.MobileNetDenseASPP import Model_CFG
        model = DenseASPP(Model_CFG, num_classes=args.num_classes)
        args.is_pytorch_model = False

    else:
        print("Model {} is not supported".format(args.model))
        print("Models can only be : \
                \n\t- DeepLabV3_Resnet50 (pytorch)\
                \n\t- DeepLabV3_Resnet101 (pytorch)\
                \n\t- FCN_Resnet50 (pytorch)\
                \n\t- FCN_Resnet101 (pytorch)\
                \n\t- UNet (external)\
                \n\t- DenseASPP121 (external)\
                \n\t- DenseASPP161 (external)\
                \n\t- DenseASPP169 (external)\
                \n\t- DenseASPP201 (external)\
                \n\t- MobileNetDenseASPP (external)")
        exit(1)

    dir = date.today().isoformat() + "-" +str(int(time.time()))
    args.save_path = os.path.join("save", args.model, dir)

    return model, args

def get_optimizer(args):
    pass
