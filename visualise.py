import cv2
import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import registers_mmyolo

registers_mmyolo.registerStuff()

# Specify the path to model config and checkpoint file
config_file = '/mnt/c/Users/chakr/Documents/Honours/thesis_ready/official_thesis_models/instance/INS-Bench/work_dirs/MoNuSeg/rtmdet/2024-02-19-02-31-27/20240219_023128.py'
checkpoint_file = '/mnt/c/Users/chakr/Documents/Honours/thesis_ready/official_thesis_models/instance/INS-Bench/work_dirs/MoNuSeg/rtmdet/2024-02-19-02-31-27/epoch_12.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# Test a single image and show the results
img = '/mnt/c/Users/chakr/Documents/Honours/thesis_ready/CURRENTLY_ON_THERMALTAKE/INSBench/INSBench/datasets/MoNuSeg/train/TCGA-18-5592-01Z-00-DX1.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# Show the results
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)