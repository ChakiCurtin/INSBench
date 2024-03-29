import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
from insbench.results import yolo_to_numpy, mask_overlay

def get_images(args: argparse.Namespace):
    images = args.dataset_root / args.dataset
    if args.normalised:
        images = images / "yolo_sn"
    elif args.originalclahe:
        images = images / "yolo_clahe"
    elif args.normalisedclahe:
        images = images / "yolo_sn_clahe"
    else:
        images /= "yolo"
    images /= "test"
    return sorted(images.glob("*.png"))

def get_mask(image_path: Path):
    image_name = image_path.name
    return image_path.parents[2] / "masks" / "test" / image_name

def get_source(args: argparse.Namespace):
    source = args.dataset_root / args.dataset
    if args.normalised:
        source = source / "yolo_sn"
    else:
        source = source / "yolo"
    source = source / "test"
    return source

def main(args: argparse.Namespace):
    # -- Load model -- #
    model = YOLO(args.model)

    # Classes and palette are predefined
    classes = ("background", "nucleus")
    palette = [[0, 0, 0], [0, 255, 0]]

    # -- Predict with model -- #
    results = model(get_source(args), max_det=args.max_det, stream=True)

    for result in results:
        img = result.orig_img
        image = Path(result.path)

        ground_truth = cv2.imread(str(get_mask(image)), cv2.IMREAD_GRAYSCALE)
        result = yolo_to_numpy(result)

        if args.binary:
            # Under the hood, this function converts the result to a numpy array and displays it
            original = cv2.equalizeHist(ground_truth)
            prediction = cv2.equalizeHist(result)
        else:
            original = mask_overlay(img, ground_truth, classes, palette, args.outline)
            prediction = mask_overlay(img, result, classes, palette, args.outline)
        
        # -- [ if save ] -- #
        if args.save is not None:
            image_name = str(image.stem) + ".png"
            full_image_path = args.save / "preds"
            if not os.path.exists(full_image_path):
                os.mkdir(full_image_path)
                print("[*]: 'preds' directory was not found, creating ")
            if args.binary:
                full_image_path = full_image_path / "binary"
            if args.outline:
                full_image_path = full_image_path / "outline"
            if not os.path.exists(full_image_path):
                os.mkdir(full_image_path)
                print("[*]: 'preds' directory was not found, creating ")
            save_name = full_image_path / image_name
            cv2.imwrite(filename=str(save_name), img=prediction)
        # -- [ if show ] -- #
        if args.show:
            cv2.imshow(f"{image.stem} - Ground Truth", original)
            cv2.imshow(f"{image.stem} - Prediction", prediction)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser("Training script for the Ultralytics YOLOv8 model")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--model", type=Path, required=True, help="The path to the trained model to use in inference")
    parser.add_argument("--max-det", type=int, default=1600, help="The maximum number of instances that can be predicted")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path to where datasets are stored")
    parser.add_argument("-b", "--binary", action='store_true', help="Whether black and white binary masks should be produced")
    parser.add_argument("--outline", action='store_true', default=True, help="Whether overlays should be drawn as an outline instead of filled in")
    parser.add_argument("--normalised", action='store_true', help="Whether to test on stain normalised images")
    parser.add_argument("--originalclahe", action="store_true", help="Whether to train on the original dataset with CLAHE applied",)
    parser.add_argument("--normalisedclahe", action="store_true", help="Whether to train on the stain normalised dataset with CLAHE applied",)
    parser.add_argument("--save", type=Path, help="The path to where predicted images are stored")
    parser.add_argument("--show", action="store_true", help="Whether to show prediction images (cv2)",)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)