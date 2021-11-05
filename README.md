# Pipeline for semantic segmentation

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This project has been developed during my internship at LISSI. The objective was to create a pipeline for semantic segmentation with python and pytorch.

## 1. Documentation
You can read the documentation [here](https://github.com/MalondaClement/pipeline/wiki).

## 2. How to use the pipeline

```bash
git clone https://github.com/MalondaClement/pipeline.git
```

### 2.1 Training
#### 2.1.1 Parameters for training
Before starting training it's possible to change training parameters using `ARGS` constructor in `train.py` script.

Parameters of the constructor :
* Models name,
* Dataset path,
* Number of classes,
* Type of labels the dataset used ("label", "csv" or "json") (__depends of the dataset__),
* Batch size,
* Number of epoch.

#### 2.1.2 Start training
```
python3 train.py
```

#### 1.3 Save directory
```
pipeline
└── save
    ├── model_name
    │   ├── yyyy-mm-dd-id
    │   │   ├── best_weights.pth.tar
    │   │   ├── checkpoint.pth.tar
    │   │   └── learning_curves.png
    │   └── yyyy-mm-dd-id
    │   │   ├── best_weights.pth.tar
    │   │   ├── checkpoint.pth.tar
    │   │   └── learning_curves.png
    └── model_name
        └── yyyy-mm-dd-id
            ├── best_weights.pth.tar
            ├── checkpoint.pth.tar
            └── learning_curves.png
```
### 2.2. Evaluation

```
python3 evaluation.py
```

### 2.3. Inference

```
python3 inference.py
```
