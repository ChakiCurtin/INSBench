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
pip install ultralytics
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
# download pre-trained yolov8x-seg file from ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt

# now train with dataset and specifying params (change according to your specs)
CUDA_VISIBLE_DEVICES='0' yolo segment train data=yamls/monuseg.yaml model=yolov8x-seg.pt epochs=550 lr0=0.0003 max_det=1000 batch=2 cache=True project=MoNuSeg name=yolov8_1xb2-500e_monuseg_OG-640x640 single_cls=True optimizer='Adam'
```
2. export_yolo.py
   1. create exports (binary images) for YOLO
   2. Note for the follwing code below:
        - "--dataset": Same as above
        - "--dataset-root": Same as above
        - "--model": path + name of weights trained for the model i.e: "/path/to/config/weights/last.pt"
        - "--save": path to directory to save exports. i.e: "/path/to/config/"
```bash
CUDA_VISIBLE_DEVICES=0 python3 export_yolo.py --dataset MoNuSeg --dataset-root "/path/to/datasets/" --model "/path/to/config/weights/last.pt" --save "/path/to/config/"
```
3. evaluate.py : SAME AS ABOVE

### EXTRA

1. clahe.py
   1. This provides the Contrast Limited Adaptive Histogram Equalisation (CLAHE) augmentation to a dataset
   2. Notes for the following command:
        - "--dataset": Same as above
        - "--dataset-root": Same as above
        - "--set": the dataset set you want thresholded (train, val, test)
        - "--threshold": The CLAHE threshold you want to set. Effective range 0.1 ~ 3.0
        - "--save": Path to save the augmented dataset
```bash
python3 clahe.py --dataset MoNuSeg --dataset-root "/path/to/dataset" --set train --threshold 2.0 --save "/path/to/save/"
```
2. manipulations.py
   1. This script allows for better visual analysis with predictions directly overlapped with ground truth. (Pixels of ground truth and prediction are in different colour and overlap showing where both meet.)
   2. Notes for the following command:
        - "--dataset": Same as above
        - "--dataset-root": Same as above
        - "--image-path": Path to prediction exports for analysis
        - "--save": Same as above
```bash
python3 manipulations.py --dataset MoNuSeg --dataset-root "/path/to/dataset" --image-path "/path/to/exports" --save "/path/to/save"
```
3. visualise_exports.sh
   1. This script allows for qualitative analysis through two options (outline and overlay). 
      1. Outline traces the boundaries of nuclei onto the test image to show a better view on how well the prediction actually performed on the image.
      2. Overlay directly places the segmentation map created by SAM or otherwise onto the test image. The default way to assess how a prediction map performs however, outline gives a better understanding for semantic segmentation (1 class when doing Cell Nuclei Analysis (CNS))
   2. Notes on the following code:
        - "--dataset": Same as above
        - "--dataset-root": Same as above
        - "--image-root": Same as above
        - "--overlay/outline": Either overlay or outline can be used at one time.
        - "--save": Same as above
```bash
CUDA_VISIBLE_DEVICES=0 python3 visualise_exports.py --dataset "MoNuSeg" --dataset-root "/path/to/dataset" --image-root "/path/to/prediction/exports" --overlay/outline --save "/path/to/save"
```
4. visualise_yolo.py
   1. Same as visualise_exports.py but only handles YOLO from Ultralytics and can only run outlines
```bash
CUDA_VISIBLE_DEVICES=0 python3 visualise_yolo.py --dataset "MoNuSeg" --dataset-root "/path/to/dataset" --model "/path/to/model/weights/last.pt" --save "/path/to/save"
```

## Uninstall

If conda has been used to install this project, easiest way will be to delete the environment and delete the repo directory. Otherwise, you will need to uninstall each dependency as specified in the install section.