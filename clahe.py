import argparse
from pathlib import Path
import shutil
import sys
import cv2

def save_results(save_path: Path, dir_name: str, clahe_img: any, txt_path: Path, save_name: str):
    save_dir = save_path / dir_name
    # -- [ Check for folder, otherwise create it ] -- #
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    total_name_res = save_dir / str(save_name)
    cv2.imwrite(str(total_name_res), clahe_img)
    shutil.copy(txt_path,save_dir)

def show_results(original: any, clahe: any):
    # -- [ Show Images ] -- #
    cv2.imshow("Original Image:", original)
    cv2.imshow("CLAHE Thresholded:", clahe)

def image_CLAHE(im_path: Path, txt_path: Path, args: argparse.Namespace, dir_name: str):
    # -- [ Set up CLAHE ] -- #
    clahe = cv2.createCLAHE(clipLimit=args.threshold, tileGridSize=(8,8)) # default tilegrid

    # -- [ Read in the image given from path ] -- #
    original = cv2.imread(str(im_path)) # BGR
    assert original is not None, "file could not be read, check with os.path.exists()"
    # -- [ Apparently, CLAHE cannot be done to RGB directly so convert to LAB and apply then revert back ] -- #
    # -- [ https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images ] -- #
    
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    #lab_planes = cv2.split(lab)

    # -- [ perform CLAHE Thresholding ] -- #
    #lab_planes[0] = clahe.apply(lab_planes[0])
    #lab = cv2.merge(lab_planes)
    #imcl = clahe.apply(original) # WHAT I DID B4
    lab[:,:,0] = clahe.apply( lab[:,:,0])
    imcl = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 

    # -- [Next step ] -- #
    if args.show:
        show_results(original=original,clahe=imcl)
    if args.save is not None:
        save_results(save_path=args.save,dir_name=dir_name,clahe_img=imcl,txt_path=txt_path, save_name=im_path.name)

def main(args: argparse.Namespace):

    # -- [ pre checks ] -- #
    if args.image_path and args.dataset_root is not None:
        sys.exit("Cannot do CLAHE for both dataset and secondary image source")  
    
    if args.dataset_root is not None and args.dataset is None:
        sys.exit("Please provide which dataset to use")  

    if args.dataset_root is None and args.dataset is not None:
        sys.exit("Cannot have a dataset without root path")  

    if args.set is None and args.dataset is not None:
        sys.exit("Please choose which folder to apply CLAHE in [train, val, test, all]")  

    images_path = None
    texts_path = None
    dir_name = None

    if args.image_path is not None:
        images_path = args.image_path
        texts_path = args.image_path
        dir_name = "Clahed"
    
    if args.dataset_root is not None:
        if args.dataset is not None:
            if args.set is not None:
                if args.set == "all":
                    sys.exit("TODO: Recreate whole dataset with CLAHE applied, for now rerun with train, val and test and recreate yourself")
                else:
                    if args.normalised:
                        images_path = args.dataset_root / args.dataset / "yolo_sn" / args.set
                        texts_path = args.dataset_root / args.dataset / "yolo_sn" / args.set
                    else:
                        images_path = args.dataset_root / args.dataset / "yolo" / args.set
                        texts_path = args.dataset_root / args.dataset / "yolo" / args.set
                    dir_name = args.set
                    

    image_paths = sorted(images_path.glob("*.png"))
    assert image_paths is not None, "Something went wrong"
    text_paths = sorted(texts_path.glob("*.txt"))
    assert text_paths is not None, "Something went wrong"
    
    print("[*] Images found: "+ str(len(image_paths)))
    print("[*] Image path: " + str(images_path))

    data = list(zip(image_paths, text_paths))
    print("[*] Total to do: "+ str(len(data)))
    for img, txt in data:
        image_CLAHE(im_path=img, txt_path=txt, args=args, dir_name=dir_name)
        if args.show:
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

def get_args() -> argparse.Namespace:
    DATASETS = ["MoNuSeg", "MoNuSAC", "CryoNuSeg", "TNBC"]
    SETS = ["train", "val", "test", "all"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, type=str, default=None, choices=DATASETS, help="The dataset folder name [MoNuSeg, MoNuSAC, CryoNuSeg, TNBC]")
    parser.add_argument("--dataset-root", type=Path, default=None, required=False, help="The path to where datasets are stored")
    parser.add_argument("--image-path", type=Path, default=None, required=False, help="Directory of Images to apply CLAHE on (if not dataset)")
    parser.add_argument("--set", required=True, type=str, choices=SETS, default="test", help="Which set the predictions are apart of: [train, val, test, all]")
    parser.add_argument("--save", type=Path, required=False, help="directory to save the resulting images to")
    parser.add_argument("--show", action='store_true', required=False, help="show CLAHE thresholded images on screen")
    parser.add_argument("--normalised", action='store_true', required=False, help="To apply CLAHE on normalised dataset (normalisation must have already been done beforehand)")
    parser.add_argument("--threshold", type=float, default=10.0, required=False, help="CLAHE threshold to apply (default = 40)")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    main(args)