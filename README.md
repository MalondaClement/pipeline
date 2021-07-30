# Pipeline for semantic segmentation

This project has been developed during my internship at LISSI. The objective was to create a pipeline for semantic segmentation with python and pytorch.


## :warning: Prevent batch size problem for training :warning:

When you choose the batch size, you need to be careful about the value :
* `args.batch_size` != 1
*  dataset size modulo `agrs.batch_size` !=1


## Path format for models saves

The train script create a new directory for each execution using the date.
* 📁 pipeline/
    * 📁 save/
        * 📁 model/
            * 📁 yyyy-mm-dd-id/
                * best_weights.pth.tar
                * checkpoint.pth.tar
                * learning_curves.png
            * 📁 yyyy-mm-dd-id/
                * ...
        * 📁 model/
            * 📁 yyyy-mm-dd-id/
                * ...

## Pipeline architecture

* 📁 pipeline/
    * 📁 helpers/
        * 📄 ARGS.py
        * 📄 helpers.py
        * 📄 labels.py
    * 📁 learning/
        * 📄 cityscapes.py
        * 📄 minicity.py
        * 📄 learner.py
        * 📄 tunnel.py
        * 📄 utils.py
    * 📁 models/
        * 📁 configs/
            * 📄 DenseASPP121.py
            * 📄 DenseASPP161.py
            * 📄 DenseASPP169.py
            * 📄 DenseASPP201.py
            * 📄 MobileNetDenseASPP.py
        * 📄 DenseASPP.py
        * 📄 MobileNetDenseASPP.py
        * 📄 UNet.py
        * 📄 utils.py
