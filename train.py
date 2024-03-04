import argparse
import datetime
from mmengine import Config
from mmengine.runner import Runner
from pathlib import Path
import registers_mmyolo

def main(args: argparse.Namespace):
    registers_mmyolo.registerStuff()
        # -- Get config files  -- #
    configs_dir = Path("configs")
    # -- Grab the config location based on arguments -- #
    #default_cfg = Config.fromfile(configs_dir / "default.py")
    model = args.model
    if model == "rtmdet":
        configs_model_dir = Path(configs_dir) / "mmdet"
        if args.normalised:
            default_cfg = Config.fromfile(configs_model_dir / "rtmdet_SN.py" )
        elif args.originalclahe:
            default_cfg = Config.fromfile(configs_model_dir / "rtmdet_OG_CLAHE.py" )
        elif args.normalisedclahe:
            default_cfg = Config.fromfile(configs_model_dir / "rtmdet_SN_CLAHE.py" )
        else:
            default_cfg = Config.fromfile(configs_model_dir / "rtmdet_OG.py")
    elif model == "mmyolo":
        configs_model_dir = Path(configs_dir) / "mmyolo"
        default_cfg = Config.fromfile(configs_model_dir / "yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py") # Model 

    #dataset_cfg = Config.fromfile(configs_dir / "default_dataset.py")       # COCO DATASET
    #default_model_cfg = Config.fromfile(configs_dir / "default_model.py")   #
    #model_cfg = Config.fromfile(configs_dir / f"{args.model.lower()}.py")  # Will change once we have more models
    #runtime_cfg = Config.fromfile(configs_dir / "default_runtime.py")       # default runtime 
    
        # -- Create the Config -- #
    cfg = Config()
    cfg.merge_from_dict(default_cfg.to_dict()) 
    #cfg.merge_from_dict(dataset_cfg.to_dict())
    #cfg.merge_from_dict(runtime_cfg.to_dict())

        # -- Fill in dataset settings -- #
    #cfg.dataset_type = f"{args.dataset}Dataset"
    cfg.dataset_type = "MoNuSegDataset"
    cfg.data_root = str(args.dataset_root / args.dataset)
    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root

    cfg.test_dataloader.dataset.type = cfg.dataset_type
    cfg.test_dataloader.dataset.data_root = cfg.data_root

    # fill in routes for dataset
    ANN_FILE_TRAIN = 'annotations/train.json'
    ANN_FILE_VAL = 'annotations/val.json'
    ANN_FILE_TEST = 'annotations/test.json'

    cfg.val_evaluator.ann_file = cfg.data_root + "/" + ANN_FILE_VAL
    cfg.test_evaluator.ann_file = cfg.data_root + "/" + ANN_FILE_TEST
    if args.normalised:
        cfg.train_dataloader.dataset.ann_file = ANN_FILE_TRAIN
        cfg.train_dataloader.dataset.data_prefix.img = 'yolo_sn/train/'
        cfg.val_dataloader.dataset.ann_file = ANN_FILE_VAL
        cfg.val_dataloader.dataset.data_prefix.img = 'yolo_sn/val/'
        cfg.test_dataloader.dataset.ann_file = ANN_FILE_TEST
        cfg.test_dataloader.dataset.data_prefix.img = 'yolo_sn/test/'


    elif args.originalclahe:
        cfg.train_dataloader.dataset.ann_file = ANN_FILE_TRAIN
        cfg.train_dataloader.dataset.data_prefix.img = 'yolo_clahe/train/'
        cfg.val_dataloader.dataset.ann_file = ANN_FILE_VAL
        cfg.val_dataloader.dataset.data_prefix.img = 'yolo_clahe/val/'
        cfg.test_dataloader.dataset.ann_file = ANN_FILE_TEST
        cfg.test_dataloader.dataset.data_prefix.img = 'yolo_clahe/test/'
    elif args.normalisedclahe:
        cfg.train_dataloader.dataset.ann_file = ANN_FILE_TRAIN
        cfg.train_dataloader.dataset.data_prefix.img = 'yolo_sn_clahe/train/'
        cfg.val_dataloader.dataset.ann_file = ANN_FILE_VAL
        cfg.val_dataloader.dataset.data_prefix.img = 'yolo_sn_clahe/val/'
        cfg.test_dataloader.dataset.ann_file = ANN_FILE_TEST
        cfg.test_dataloader.dataset.data_prefix.img = 'yolo_sn_clahe/test/'
    else:
        cfg.train_dataloader.dataset.ann_file = ANN_FILE_TRAIN
        cfg.train_dataloader.dataset.data_prefix.img = 'yolo/train/'
        cfg.val_dataloader.dataset.ann_file = ANN_FILE_VAL
        cfg.val_dataloader.dataset.data_prefix.img = 'yolo/val/' 
        cfg.test_dataloader.dataset.ann_file = ANN_FILE_TEST
        cfg.test_dataloader.dataset.data_prefix.img = 'yolo/test/'

    # for memory 
    cfg.train_dataloader.persistent_workers = False
    cfg.test_dataloader.persistent_workers = False
    cfg.val_dataloader.persistent_workers = False
        # -- set batch size for model -- #
    cfg.train_dataloader.batch_size = args.batch
    cfg.test_dataloader.batch_size = args.batch
    cfg.val_dataloader.batch_size = args.batch

    cfg.train_cfg.max_epochs = args.epochs

    cfg.metainfo = {
        'classes': ('nucleus', ), # dataset category name
        'palette': [
            (220, 20, 60),
        ]}
    
    # CHANGE DEFAULT SCOPE FOR RUNTIME
    cfg.default_scope ='mmyolo'

        # -- Create the work directory -- #
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H_%M_%S")
    # work_dir = f"./work_dirs/{args.dataset}/{args.model}/{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}/"
    # cfg.work_dir = work_dir
    work_dir = f"./work_dirs/{args.dataset}/{args.model}/"
    if args.normalised:
        work_dir += "SN_"
    if args.originalclahe:
        work_dir += "OG_CLAHE_"
    if args.normalisedclahe:
        work_dir += "SN_CLAHE_"
    work_dir += f"{now_str}/"
    cfg.work_dir = work_dir

    # -- SETTING DEFAULT HOOKS AS PER DOCS FOR NOW (copy of what is in default_runtime.py) -- #

    # default_hooks = dict(
    #     timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    #     logger=dict(type='LoggerHook', interval=10),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    #     param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    #     checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
    #     sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
    #     visualization=dict(type='mmdet.DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results

    default_hooks = dict(
            timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
            logger=dict(type='LoggerHook', interval=1),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
            param_scheduler=dict(type='YOLOv5ParamSchedulerHook',
                                max_epochs=args.epochs,
                                scheduler_type='linear',
                                lr_factor=0.01,
                                ), # update some hyper-parameters of optimizer
            checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,save_last=True,), # Save checkpoints periodically
            sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
            visualization=dict(type='mmdet.DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results


    cfg.default_hooks = default_hooks

    # -- Set seed to facilitate reproducing the result -- #
    cfg['randomness'] = dict(seed=0)

    # -- Let's have a look at the final config used for training -- #
    print(f'Config:\n{cfg.pretty_text}')

    # -- Train a model -- #
    runner = Runner.from_cfg(cfg)
    runner.train()

def get_args() -> argparse.Namespace:
    #DATASETS = ["MoNuSeg", "MoNuSAC", "TNBC", "CryoNuSeg"]
    DATASETS = ["MoNuSeg"]
    MODELS = ["rtmdet","mmyolo"]

    parser = argparse.ArgumentParser("Training script for training models on the mmsegmentation architecture")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="The model to use for training")
    parser.add_argument("--batch", type=int, default=2, help="The batch size to use during training")
    parser.add_argument("--wandb", action="store_true", help="Enables Weights and Biases for logging results")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="The dataset to use for training")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"), help="The base directory for datasets")
    parser.add_argument("--epochs", type=int, default=500, help="The maximum epochs for training")
    parser.add_argument("--val-interval", type=int, default=1000, help="Validation interval during training in iterations")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging interval during training in iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, help="Checkpointing interval in iterations")
    parser.add_argument("--normalised",action="store_true",help="Whether to train on stain normalised images",)
    parser.add_argument("--originalclahe",action="store_true",help="Whether to train on the original dataset with CLAHE applied",)
    parser.add_argument("--normalisedclahe",action="store_true",help="Whether to train on the stain normalised dataset with CLAHE applied",)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # Ensure old profiling data is cleaned up
    main(args)