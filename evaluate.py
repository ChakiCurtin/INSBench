import argparse
from pathlib import Path
import numpy as np
from insbench import metrics, istarmap
from multiprocessing import Pool
import pandas as pd
import cv2
from tqdm import tqdm


def main(args: argparse.Namespace):
    maskdir = get_maskdir(args)
    gt_paths = sorted(maskdir.glob("*.png"))
    pred_paths = sorted(args.compare_root.glob("*.png"))

    results = {}

    pooldata = list(zip(gt_paths, pred_paths))

    with Pool(4) as pool:
        comparisons = [
            comparison
            for comparison in tqdm(
                pool.istarmap(compare_masks, pooldata), total=len(pooldata)
            )
        ]
        # comparisons = pool.starmap(compare_masks, pooldata)

    for comparison in comparisons:
        for key, value in comparison.items():
            if key not in results:
                results[key] = []
            results[key].append(value)

    df = pd.DataFrame.from_dict(results)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.set_index("name", inplace=True)

    print(df.groupby("split").median().loc[:, "f1":"hausdorff"].round(3))
    # print(df.groupby("split").mean().loc[:, "f1":"hausdorff"].round(3))


def compare_masks(gt_path: Path, pred_path: Path):
    if gt_path.name != pred_path.name:
        print("Woah there buddy something seems to have gone awry")

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if args.sam:
        _, gt = cv2.threshold(gt,0,255,cv2.THRESH_BINARY)
        _, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)

    # cv2.imshow(f"{pred_path.name} - Prediction", gt)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # if key == ord('q'):
    #     exit()

    comparisons = {}
    comparisons["name"] = gt_path.stem

    parts = gt_path.parts
    if "train" in parts:
        comparisons["split"] = "train"
    elif "val" in parts:
        comparisons["split"] = "val"
    elif "test" in parts:
        comparisons["split"] = "test"

    if args.sam:
        comparisons["accuracy"] = metrics.accuracy(gt, pred, class_idx=255)
        comparisons["precision"] = metrics.precision(gt, pred, class_idx=255)
        comparisons["recall"] = metrics.recall(gt, pred, class_idx=255)
        comparisons["f1"] = metrics.f1(comparisons["precision"], comparisons["recall"])
        comparisons["iou"] = metrics.iou(gt, pred, class_idx=255)
        comparisons["hausdorff"] = metrics.general_object_hausdorff(gt, pred)
    else:
        comparisons["accuracy"] = metrics.accuracy(gt, pred)
        comparisons["precision"] = metrics.precision(gt, pred)
        comparisons["recall"] = metrics.recall(gt, pred)
        comparisons["f1"] = metrics.f1(comparisons["precision"], comparisons["recall"])
        comparisons["iou"] = metrics.iou(gt, pred)
        comparisons["hausdorff"] = metrics.general_object_hausdorff(gt, pred)

    return comparisons


def get_maskdir(args: argparse.Namespace):
    maskdir = args.dataset_root / args.dataset / "masks" / "test"
    return Path(maskdir)


def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        required=True,
        help="The dataset to use when comparing masks",
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        required=True,
        help="The path where predictions lie",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help="The path where datasets are stored",
    )
    parser.add_argument("--sam", action="store_true", help="Whether the exports come from SAM or not ",)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Ensure old profiling data is cleaned up
    main(args)
