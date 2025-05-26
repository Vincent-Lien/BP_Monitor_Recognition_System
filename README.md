# BP_Monitor_Recognition_System
圖形識別期末專題：血壓記錄小幫手

## Setup
The repo is run on Python 3.12.3  
1. You can install the libraries by
```
pip install -r requirements.txt
```
2. If you are using conda
```
conda create -n bp_monitor python=3.12.3
conda activate bp_monitor
pip install -r requirements.txt
```

## Run directly

### Download Checkpoints
1. Download the checkpoints from this link: 
2. Put the checkpoints into `checkpoints`
```
mkdir checkpoints
```
Expect
```
BP_Monitor_Recognition_System
├── checkpoints
│   ├── localization_best.pt
│   └── seven_seg_classification_best.pth
```

### Demo
Run demo with gradio
```
python bp_app.py
```

## Train by yourself

### Datasets
Get the training datasets from the urls:
- Localization: https://universe.roboflow.com/sphygmomanometer/sphygmomanometer-qcpzd/dataset/10  
- Seven-segment single numbers classification: https://www.kaggle.com/datasets/testtor/sevensegment-numbers?
And put them into `dataset`
```
mkdir dataset
```
expect
```
BP_Monitor_Recognition_System
├── dataset
│   ├── localization
│   │   ├── test
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   └── valid
│   │       ├── images
│   │       └── labels
│   └── seven_seg_classify
│       ├── 0
│       ├── 1
│       ├── 2
│       ├── 3
│       ├── 4
│       ├── 5
│       ├── 6
│       ├── 7
│       ├── 8
│       └── 9
```

### Train
1. Train localizer (yolo)
```
python train/train_localization.py
```
2. move the `runs/detect/train/weights/best.pt` into `checkpoints` and rename `localization_best.pt`
```
mkdir checkpoints
mv runs/detect/train/weights/best.pt checkpoints/localization_best.pt
```
3. Train classifier (ResNet)
```
python train/train_seven_seg_classification.py --data_dir dataset/seven_seg_classify/
```

### Inference
Inference directly
```
python inference.py
```
Or you can also demo on `bp_app.py`
```
python bp_app.py
```