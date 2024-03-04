import cv2
import mmcv
from mmyolo.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from mmseg.structures import SegDataSample
import registers_mmyolo
import argparse
from pathlib import Path
import glob
import os
import torch
import numpy as np
import torchvision.transforms as T
# FOR SAM
from segment_anything import sam_model_registry, SamPredictor

# def old_visualier_code():
#      # Specify the path to model config and checkpoint file
#     config_file = ""
#     checkpoint_file = ""
#     # Build the model from a config file and a checkpoint file
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     # Init visualizer
#     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#     # The dataset_meta is loaded from the checkpoint and
#     # then pass to the model in init_detector
#     visualizer.dataset_meta = model.dataset_meta
#     # Test a single image and show the results
#     img = ""  # or img = mmcv.imread(img), which will only load it once
#     result = inference_detector(model, img)
#     # Show the results
#     img = mmcv.imread(img)
#     img = mmcv.imconvert(img, 'bgr', 'rgb')
#     visualizer.add_datasample(
#         'result',
#         img,
#         data_sample=result,
#         draw_gt=False,
#         show=True)

# -- [ TAKEN FROM UTILS IN STREAMLIT APPLICATION AND MODIFIED ] -- #
def inference_detections(model, image):
    #print("[*] Generating result for image: " + image.name)
    result = inference_detector(model, image)
    nuclei_list = []
    print(f"NUMBER OF NMS SET: {model.cfg.model_test_cfg.max_per_img}")
    print(f"NUMBER OF NMS FOUND: {len(result.pred_instances.bboxes)}")
    print("Using the lower value")
    number = 0
    if model.cfg.model_test_cfg.max_per_img > len(result.pred_instances.bboxes):
        number = len(result.pred_instances.bboxes)
    else:
        number = model.cfg.model_test_cfg.max_per_img
    for xx in range(number):
        X1 = int(result.pred_instances.bboxes[xx][0])
        Y1 = int(result.pred_instances.bboxes[xx][1])
        X2 = int(result.pred_instances.bboxes[xx][2])
        Y2 = int(result.pred_instances.bboxes[xx][3])

        nuclei_list.append([X1,Y1,X2,Y2])
    return nuclei_list

def show_box_cv(box_s, img):
    for box in box_s:
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
    return img

# -- [ MMYOLO STUFF ] -- #
def mmyolo_init(config, pathfile):
    print("[*] Loading path file...")
    print("[*] Loading config file...")
    print("[*] Building model..")
    model = init_detector(str(config), str(pathfile), device='cuda:0')
    print("[*] Initialising Visualiser..")
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model

# -- [ SAM STUFF ] -- #
def sam_init(args):
    sam_checkpoint = args.sam
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def input_boxes_sam(nuclei_list):
    # for 721 boxes
    inputs_boxes = []
    boxes = int(len(nuclei_list) / 4)
    remaining = len(nuclei_list) % 4

    print(f"Splitting input boxes into 4 with the remainding as the fifth: number each split: {boxes}. Number in last split: {remaining}")
    y = 0
    z = boxes   
    for _ in range(4):
        inputs_boxes.append(nuclei_list[y:z])
        y += boxes
        z += boxes
    if remaining != 0:
        inputs_boxes.append(nuclei_list[(boxes * 4):(boxes * 4 + remaining)])
    #inputs_boxes = [nuclei_list[0:199],nuclei_list[200:399],nuclei_list[400:599],nuclei_list[600:799],nuclei_list[800:999]]
    return inputs_boxes

def masks_array_sam(masks_list, random_color=False):
    masked_out_list = []
    for masks in masks_list:
        #print("[**] Plotting batch masks | batch: " + str(ii))
        sub_mask_list = []
        for mask in masks:
            mask = mask.cpu().numpy()
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            sub_mask_list.append(mask_image)
        #print("[**] saving batch of masks | batch: " + str(ii))
        batched_mask = sub_mask_list[0]
        for iii in range(len(sub_mask_list)):
            batched_mask = cv2.bitwise_or(batched_mask, sub_mask_list[iii])
        masked_out_list.append(batched_mask)
    batched_mask = masked_out_list[0]
    for ii in range(len(masked_out_list)):
        batched_mask = cv2.bitwise_or(batched_mask, masked_out_list[ii])
    return batched_mask

def prediction_masks_sam(image, predictor, inputs_boxes):
    masks_list = []
    predictor.set_image(image)
    for section in inputs_boxes:
        input_box = np.array(section)
        input_box = torch.tensor(input_box, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        masks_list.append(masks)
    return masks_list

# -- [ PIPELINE STUFF ] - #
def process_image_pipeline(model, predictor, path_img, image, args):
    # -- Inference detection -- #
    detections = inference_detections(model=model, image=path_img) 
    # -- process inference to inputs for SAM -- #
    inputs_boxes = input_boxes_sam(detections)
    # -- get prediction information from SAM -- #
    masks_list = prediction_masks_sam(image=image, predictor=predictor, inputs_boxes=inputs_boxes)
    # -- process these masks into one image array -- #
    batched_mask = masks_array_sam(masks_list=masks_list, random_color=args.colour)
    # -- add the mask to current session-- #
    return detections, batched_mask
    #st.session_state.detections = detections
    #st.session_state.batched_mask = batched_mask

# -- [ GENERAL STUFF ] -- #
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


def main(args: argparse.Namespace):
    registers_mmyolo.registerStuff()
    images = get_images(args=args)
    if len(images) == 0:
        print("No images found!")
        return
    # INITS
    model = mmyolo_init(config=args.config,pathfile=args.pth)
    predictor = sam_init(args)
    for image in images:
        print(image)
        img = cv2.imread(str(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"[*]: for Image: {image.stem}")
        #detections, batched_mask = process_image_pipeline(model=model, predictor=predictor, path_img=image, image=img, args=args)
        _, batched_mask = process_image_pipeline(model=model, predictor=predictor, path_img=image, image=img, args=args)

        if args.export and args.save:
            print(f"[*] EXPORT: exporting {image.stem} predictions")
            save_dir = args.save / "export" 
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                print("[*]: 'export' directory was not found, creating ")
            save_name = save_dir / f"{image.stem}.png"
            # make image saveable
            export_mask = batched_mask * 255
            export_mask = export_mask.squeeze()
            export_mask = export_mask.astype(np.uint8)
            #export_mask = cv2.cvtColor(batched_mask.astype(np.float32), cv2.COLOR_RGBA2RGB)
            cv2.imwrite(filename=str(save_name), img=export_mask)
        if args.show:
            cv2.imshow(f"{image.stem} - Ground Truth", img)
            cv2.imshow(f"{image.stem} - Prediction", batched_mask)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break



def get_args() -> argparse.Namespace:
    #DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    DATASETS = ["MoNuSeg"]
    parser = argparse.ArgumentParser("Instance segmentation and object detector -> pipeline detections and images extractor script")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    #parser.add_argument("--model", type=str, required=True, choices=MODELS, help="The model to use for training")
    parser.add_argument("--dataset-root", type=Path, required=True, default=Path("datasets"), help="The base directory for datasets")
    parser.add_argument("--normalised", action="store_true", help="Whether to train on stain normalised images",)
    parser.add_argument("--originalclahe", action="store_true", help="Whether to train on the original dataset with CLAHE applied",)
    parser.add_argument("--normalisedclahe", action="store_true", help="Whether to train on the stain normalised dataset with CLAHE applied",)
    # path and config files
    parser.add_argument("--config", type=Path,  help="MMYOLO Config file path (.py)")
    parser.add_argument("--pth", type=Path,  help="MMYOLO path file path (.pth)")
    parser.add_argument("--export", action="store_true",  help="export predictions from pipeline | save must also be triggered")
    parser.add_argument("--save", type=Path,  help="Where to save files")
    parser.add_argument("--show", action="store_true",  help="show predictions from pipeline")
    parser.add_argument("--colour", action="store_true",  help="show predictions from pipeline in different colours")
    parser.add_argument("--sam", type=Path, required=True ,help="Path to SAM PATH file")
    args = parser.parse_args()
    return args
   

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)