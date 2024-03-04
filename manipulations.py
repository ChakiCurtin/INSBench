import argparse
from pathlib import Path
import numpy as np
import cv2
import os

def save_results(save_path: Path, color_gt: any, color_pred: any, color_result: any, save_name: str, args: argparse.Namespace):
    save_dir = save_path / "overlap"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("[*]: 'overlap' folder not found in save path, creating..")

    total_name_gt = save_dir / str("gt_" + save_name)
    total_name_pred = save_dir / str("pred_" + save_name)
    total_name_res = save_dir / str("res_" + save_name)

    # SAVING IMAGES
    if args.just_overlap:
        cv2.imwrite(str(total_name_res), color_result)
    else:
        cv2.imwrite(str(total_name_res), color_result)
        cv2.imwrite(str(total_name_gt), color_gt)
        cv2.imwrite(str(total_name_pred), color_pred)
    

def calc_overlap(clahe_gt: any, clahe_pred: any, name: str, file: str, save:bool):

    just_overlap = cv2.bitwise_and(clahe_gt, clahe_pred) # Only the overlap region between 
    num_overlap = np.count_nonzero(just_overlap)
    num_gt = np.count_nonzero(clahe_gt)
    # We have number of pixels overlapped and number of pixels in ground truth
    # Accuracy is overlap / gt
    accuracy = num_overlap / num_gt

    name_str = "Image name: " + str(name)
    gt_str = "Number of non-zero pixels in ground truth: " + str(num_gt)
    overlap_str = "Number of non-zero pixels in overlapping region (exclusive): " + str(num_overlap)
    acc_str = "Accuracy of Model ( overlap / gt ): " + str(round((accuracy * 100), 3)) + "%"
    border = "--------------------------------------------------------------------------------------"
    print(border)
    print(name_str)
    print(gt_str)
    print(overlap_str)
    print(acc_str)
    print(border)

    # -- if saving to file -- #
    if save:
        # -- Open file for ammending -- #
        file = file / "eval.txt"
        f = open(file=file, mode="a+")

        # -- ADD these results to file -- #
        f.write(border + "\n")
        f.write(name_str + "\n")
        f.write(gt_str + "\n")
        f.write(overlap_str + "\n")
        f.write(acc_str + "\n")
        f.write(border + "\n")

        # -- close the writestream -- #
        f.close()

    #return num_gt

def show_results(color_gt: any, color_pred: any, color_result: any, args: argparse.Namespace):
    # -- [ Show Images ] -- #

    if args.just_overlap:
        cv2.imshow("Overlap ", color_result)
    else:
        cv2.imshow("Prediction coloured ", color_pred)
        cv2.imshow("Ground truth coloured ", color_gt)
        cv2.imshow("Overlap ", color_result)
    

def get_maskdir(args: argparse.Namespace):
    maskdir = args.dataset_root / args.dataset / "masks" / args.set
    return Path(maskdir)

def mask_overlap(gt_path: Path, pred_path: Path, args: argparse.Namespace):

    # -- Create the save location (if provided in args) -- #
    save_dir = pred_path # TEMP
    save_q = False
    if args.save is not None:
        save_q = True
        save_dir = args.save / "overlap_visuals"
        # -- [ Check for folder, otherwise create it ] -- #
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

    if gt_path.name != pred_path.name:
        print("Woah there buddy something seems to have gone awry")

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    
    #TODO: make a new toggle for export images 
    _,gt_mask = cv2.threshold(gt,0,255,cv2.THRESH_BINARY)
    _,pred_mask = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)

    # -- [ adding colour to binary images ] -- #
    green = [0, 255, 0]
    blue =  [0, 0, 255]
    yel =   [255, 255, 0]
    red =   [255, 0, 0] 
    white = [255, 255, 255]

    # -- [ Ground truth mask from binary to bgr ] -- #
    gt_mask_3d = np.zeros( ( np.array(gt_mask).shape[0], np.array(gt_mask).shape[1], 3 ), dtype="uint8")    
    gt_mask_3d[:,:,0] = gt_mask
    gt_mask_3d[:,:,1] = gt_mask
    gt_mask_3d[:,:,2] = gt_mask

    # -- [ prediction mask from binary to bgr ] -- #
    pred_mask_3d = np.zeros( ( np.array(pred_mask).shape[0], np.array(pred_mask).shape[1], 3 ), dtype="uint8")    
    pred_mask_3d[:,:,0] = pred_mask
    pred_mask_3d[:,:,1] = pred_mask
    pred_mask_3d[:,:,2] = pred_mask

    # -- Will calculate overlap amount and output to file -- #
    calc_overlap(clahe_gt=gt_mask_3d, clahe_pred=pred_mask_3d, name=gt_path.name, file=save_dir, save=save_q)

    # -- [ Colouring all white pixels to another colour] -- #
    pred_mask_3d[np.where((pred_mask_3d==[255,255,255]).all(axis=2))] = red # Prediction mask
    gt_mask_3d[np.where((gt_mask_3d==[255,255,255]).all(axis=2))] = blue    # Ground truth mask
    
    # -- [ Converting all image masks made above to RGB ] -- #
    pred_mask_3d = cv2.cvtColor(pred_mask_3d, cv2.COLOR_BGR2RGB)
    gt_mask_3d = cv2.cvtColor(gt_mask_3d, cv2.COLOR_BGR2RGB)

    # -- [ Perform the subtraction ] -- #
    overlap = cv2.bitwise_xor(gt_mask_3d,pred_mask_3d)

    # -- [Next step ] -- #
    if args.show:
        show_results(gt_mask_3d,pred_mask_3d,overlap, args=args)
    if args.save is not None:
        save_results(save_dir,gt_mask_3d,pred_mask_3d,overlap,gt_path.name, args=args)

def main(args: argparse.Namespace):
    # -- [ Load the datasets masks ] -- #
    maskdir = get_maskdir(args)
    gt_paths = sorted(maskdir.glob("*.png")) # ground truth masks
    pred_paths = sorted(args.image_path.glob("*.png")) # predictions masks
    
    print("[*] Dataset set chosen: " + args.set)
    print("[*] Masks found in Dataset folder: "+ str(len(gt_paths)))
    print("[*] Masks found in prediction folder: "+ str(len(pred_paths)))

    data = list(zip(gt_paths, pred_paths))
    print("[*] total to evaluate: "+ str(len(data)))
    for gt, pred_mask in data:
        mask_overlap(gt, pred_mask, args)
        if args.show:
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]
    SETS = ["train", "val", "test"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, choices=DATASETS, help="The dataset folder name [ MoNuSeg, MoNuSAC, CryoNuSeg, TNBC ]")
    parser.add_argument("--dataset-root", type=Path, default=None, required=True, help="The path to where datasets are stored")
    parser.add_argument("--image-path", type=Path, default=None, required=True, help="Directory of prediction binary images")
    parser.add_argument("--set", required=True, type=str, choices=SETS, default="test", help="Which set the predictions are apart of: [train, val, test]")
    parser.add_argument("--save", type=Path, required=False, help="directory to save the resulting images to")
    parser.add_argument("--show", action='store_true', required=False, help="show prediction data on screen")
    parser.add_argument("--just-overlap", action='store_true', required=False, help="show prediction data on screen")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    main(args)