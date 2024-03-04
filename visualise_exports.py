import argparse
from pathlib import Path
import cv2
import os
import numpy as np

def get_images(args: argparse.Namespace):
    images = args.image_root
    return sorted(images.glob("*.png"))

def get_original(image_path:Path, args: argparse.Namespace):
    images = args.dataset_root / args.dataset
    image_name = image_path.name
    if args.normalised:
        images = images / "yolo_sn"
    elif args.originalclahe:
        images = images / "yolo_clahe"
    elif args.normalisedclahe:
        images = images / "yolo_sn_clahe"
    else:
        images /= "yolo"
    images /= "test"
    return images / image_name

def outline(result:str):
    img = cv2.imread(str(result), cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
    print(f"[*] {result.name} | OUTLINE")
    image_mask = cv2.imread(str(get_original(result, args=args)))
    dest = image_mask.copy()
    colour_mask = np.zeros_like(image_mask)
    contours = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    colour_mask = cv2.drawContours(colour_mask, contours, -1, [0, 255, 0], thickness=2)
    binary_mask = np.nonzero(colour_mask)
    alpha = 1
    dest[binary_mask] = cv2.addWeighted(image_mask, 1 - alpha, colour_mask, alpha, 0)[binary_mask]
    return dest

def overlay(result:str):
    #TODO: Make option when dealing with sam or just yolo. 8-bit not working for cv2.imread
    img = cv2.imread(str(result), cv2.IMREAD_COLOR)
    image_mask = cv2.imread(str(get_original(result, args=args)))
    print(f"[*] {result.name} | OVERLAY")
    dest = cv2.addWeighted(img, 1, image_mask, 0.9, 0)
    return dest

def main(args: argparse.Namespace):
    results = get_images(args=args)
    print(f"[*] Number of Images found: {len(results)}")

    for result in results:
        if args.outline:
            dest = outline(result)
        if args.overlay:
            dest = overlay(result)
        # -- [ if save ] -- #
        if args.save is not None:
            image_name = str(result.stem) + ".png"
            full_image_path = args.save / "preds"
            if not os.path.exists(full_image_path):
                os.mkdir(full_image_path)
                print("[*]: 'preds' directory was not found, creating ")
            if args.outline:
                full_image_path = full_image_path / "outline"
                if not os.path.exists(full_image_path):
                    os.mkdir(full_image_path)
                    print("[*]: 'outline' directory was not found, creating ")
            if args.overlay:
                full_image_path = full_image_path / "overlay"
                if not os.path.exists(full_image_path):
                    os.mkdir(full_image_path)
                    print("[*]: 'overlay' directory was not found, creating ")
            save_name = full_image_path / image_name
            cv2.imwrite(filename=str(save_name), img=dest)
        # -- [ if show ] -- #
        if args.show:
            cv2.imshow(f"{result.name} - Prediction", dest)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    parser = argparse.ArgumentParser("Training script for the Ultralytics YOLOv8 model")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The path to where datasets are stored")
    parser.add_argument("--image-root", type=Path, default=Path("datasets"), help="The path to where export images are stored")
    parser.add_argument("--outline", action='store_true', default=False, help="Whether prediction should be drawn as an outline instead of filled in")
    parser.add_argument("--overlay", action='store_true', default=False, help="Whether the export should be just overlayed on the associated image")
    parser.add_argument("--save", type=Path, help="The path to where predicted images are stored")
    parser.add_argument("--show", action="store_true", help="Whether to show prediction images (cv2)",)
    parser.add_argument("--normalised", action='store_true', help="Whether to test on stain normalised images")
    parser.add_argument("--originalclahe", action="store_true", help="Whether to train on the original dataset with CLAHE applied",)
    parser.add_argument("--normalisedclahe", action="store_true", help="Whether to train on the stain normalised dataset with CLAHE applied",)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)