# Instance segmentation Benchmarking for Cell Nuclei Analysis (INSBench)

Tested with python 3.8.18, Ubuntu 22.04.3 (WSL2)

## Features

- Utilises the Modular framework of MMDetection and MMYolo

- Evaluates both Object detection models RTMDet and MMYOLOv8 (from MMYOLO) and standalone instance model Yolov8 (from Ultralytics)

- Added support for Instance segmentation pipeline with the Segment Anything Model (SAM) (uses object detector -> SAM pipeline)

## Installation

1. Best practices to install dependencies using some form of virtual environment. (Conda or venv)
   - MiniConda is recommended [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create the Conda environment:
```bash
conda create --name insbench python=3.8.18 -y
conda activate insbench
```
3. Pytorch must be installed
   - [Pytorch](https://pytorch.org/get-started/locally/) - Install according to the instructions

4. Install a few depencencies:
```bash
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0"
```
5. Install MMDetection and MMYOLO
   - [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
   - [MMYOLO](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html)
```bash
mim install "mmdet>=3.0.0,<4.0.0"
mim install "mmyolo"
```
6. (**OPTIONAL**) Install YOLOv8 by Ultralytics (Do this if you also want to train and evaluate the standalone instance segmentation architecture YOLOv8)
```bash

```
7. Install left over dependencies
```bash
# Find left over dependencies
```
1. For pipeline evaluation, download [SAM](https://github.com/facebookresearch/segment-anything)
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

# now find which model checkpoint to use (we use vit_h)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
```
## Usage
Each script provided handles a specific step in the process of cell segmentation for different models under instance segmentation / object detection.

### MMYOLO & MMDET (MMYOLOv8 and RTMDet)
1. train.py
   1. For training the object detector and standalone instance segmentation models
   2. Download configuration files from MMYOLO and place in the directory configs/mmyolo/
      1. Currently, MMYOLOv5 and MMYOLOv8 (OpenMMlab's versions of the YOLOv5 and YOLOv8 architecture by Ultralytics)
   3. Currently config/rtmdet works but uses MMYOLO's version of the RTMDet model. (significant changes to allow for better training)
   4. Can handle augmented variants of datasets (MoNuSeg is used). Edit the following:
      1. All instances of "data_root" in the mmyolo directory .py files
      2. If configs are standalone (like RTMDet under configs/mmdet), create the variants of that as required with correct paths to the variants of the dataset. (My Case, I have Original, stain normalised, original + CLAHE, stain norm + CLAHE)
      3. Dataset must have coco style annotation format. If your dataset has Yolo style annotation format, you can convert using my other repo [YOLO_SEG_to_COCO_Converter](https://github.com/ChakiCurtin/YOLO-SEG-to-COCO-Converter)
   5. Note for the following code below:
        - "--dataset" : The name of the dataset. if it is MoNuSeg, then the input is "MoNuSeg"
        - "--dataset-root" : The root of the dataset directory. Does not include the name of the dataset. So for dataset (MoNuSeg) path="/path/to/dataset/MoNuSeg", the input is "/path/to/dataset/"
```bash
# To train on one GPU and on the configs I have (MMYOLOv8 and RTMDet)
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset "datasetname" --model mmyolo --dataset-root "/path/to/dataset/" --batch 2 --epochs 550
```

2. pipeline_extractor.py
   1. Since the training was for the object detector, we need to use SAM to produce export masks (for evaluation and quantitative results)
   2. Note for the following code below:
        - "--dataset" : Same as above
        - "--dataset-root" : Same as above
        - "--config" : Path to config file (.py). i.e: "/path/to/config/config.py"
        - "--pth" : Path to the weights file (.pth). i.e: "/path/to/config/weights.pth"
        - "--export" : Toggles the creation of the export images based on the test set of the dataset
        - "--save" : Path for saving the exports. Usually the same directory as the config and weights files. i.e: "/path/to/config/"
        - "--colour" : Toggles coloured masks (makes sense for SAM as each input from the object detecotor is technically one mask created from SAM. Not binary like for semantic segmentation)
```bash
CUDA_VISIBLE_DEVICES=0 python3 pipeline_extractor.py --dataset "datasetname" --dataset-root "/path/to/dataset/" --config "/path/to/config/config.py" --pth "/path/to/config/weights.pth" --export --save "/path/to/config/" --colour
```

3. evaluate.py
   1. Now that we have the exports folder of evaluation images from the test set. We can assess how well the model did on the dataset.
   2. Note for the following code below:
        - "--dataset" : Same as above
        - "--compare-root" : Path to where the exports are
        - "--dataset-root" : Same as the above
        - "--sam" : Toggle to let script know these exports come from the segment anything model
```bash
python3 evaluate.py --dataset "datasetname" --compare-root "/path/to/config/export/" --dataset-root "/path/to/dataset/" --sam
```

### YOLOv8 (FROM ULTRALYTICS)

1. Since training for yolov8 is very simple with the ultralytics package. This is the code:
```bash

```

2. export_yolo.py
   1. fefe
```bash

```

3. evaluate.py



## EXTRA

1. clahe.py
   1. This provides the Contrast Limited Adaptive Histogram Equalisation (CLAHE) augmentation to a dataset


2. manipulations.py
