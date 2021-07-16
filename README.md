# Pipeline for semantic segmentation

This project has been developed during my internship at LISSI. The objective was to create a pipeline for semantic segmentation with python and pytorch.

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
