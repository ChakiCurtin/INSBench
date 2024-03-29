import argparse
from ultralytics import YOLO
from pathlib import Path
import torch
import cv2
import torchvision.transforms as T

def get_source(args: argparse.Namespace):
    source = args.dataset_root / args.dataset
    if args.normalised:
        source = source / "yolo_sn"
    elif args.originalclahe:
        source = source / "yolo_clahe"
    elif args.normalisedclahe:
        source = source / "yolo_sn_clahe"
    else:
        source /= "yolo"
    source = source / "test"
    return source

def get_save_path(args: argparse.Namespace):
    save_path = args.save / "export"
    return save_path

def main(args: argparse.Namespace):
    # -- Load model -- #
    model = YOLO(args.model)
    
    # -- Predict with model -- #
    results = model(get_source(args), max_det=args.max_det, stream=True)

    # -- Prepare save directory -- #
    save_path = get_save_path(args)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for result in results:
        name = Path(result.path).name
        height, width = result.orig_img.shape[:2]
        
        if result.masks is not None:
            masks = result.masks.data
            nuclei_mask = torch.any(masks, dim=0).to(torch.uint8)
            nuclei_mask = nuclei_mask.reshape(1, 1, *nuclei_mask.shape)

            # Tested all interpolation modes, Nearest neighbour best represents prediction (at least for square images)
            nuclei_mask = T.Resize(size=(height, width), interpolation=T.InterpolationMode.NEAREST, antialias=False)(nuclei_mask).squeeze()
        else:
            nuclei_mask = torch.zeros((height, width))
        
        image_path = str(save_path / name)
        cv2.imwrite(image_path, nuclei_mask.cpu().numpy())

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser("Training script for the Ultralytics YOLOv8 model")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--model", type=Path, required=True, help="The path to the trained model to use in inference")
    parser.add_argument("--max-det", type=int, default=1600, help="The maximum number of instances that can be predicted")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path to where datasets are stored")
    parser.add_argument("--save", type=Path, default=Path("yolov8_work_dirs"), help="The base directory for saving yolov8 training results")
    parser.add_argument("--normalised", action='store_true', help="Whether to train on stain normalised images")
    parser.add_argument("--originalclahe", action="store_true", help="Whether to train on the original dataset with CLAHE applied",)
    parser.add_argument("--normalisedclahe", action="store_true", help="Whether to train on the stain normalised dataset with CLAHE applied",)

    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)
