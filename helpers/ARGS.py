#
#  helpers/ARGS.py
#
#  Created by Clément Malonda on 13/07/2021.

class ARGS:
  def __init__(self):
    self.dataset_path = "minicity"
    self.batch_size = 2
    self.pin_memory = True
    self.num_workers = 8
    self.colorjitter_factor = 0.3
    self.train_size = [1024, 2048]
    self.test_size = [1024, 2048]
    self.crop_size = [576, 1152]
    self.dataset_mean = [0.485, 0.456, 0.406]
    self.dataset_std = [0.229, 0.224, 0.225]
    self.loss = "ce"
    self.model = ""
    self.copyblob = True
    self.cutmix = True
    self.epochs = 4
    self.save_path = ""
    self.is_pytorch_model = True
    self.num_classes = 30
