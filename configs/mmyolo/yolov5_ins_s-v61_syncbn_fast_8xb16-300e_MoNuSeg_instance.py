_base_ = 'yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py'  # noqa

data_root = '' # dataset root
# Training set annotation file of json path
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train/'  # Dataset prefix
# Validation set annotation file of json path
val_ann_file = 'annotations/val.json'
val_data_prefix = 'val/'

test_ann_file = 'annotations/test.json'
test_data_prefix = "test/"

max_epochs = 800  # Maximum training epochs 

lr_factor = 0.01  # Learning rate scaling factor

save_checkpoint_intervals = 10

metainfo = {
    'classes': ('nucleus', ), # dataset category name
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1
# Set batch size to 4
train_batch_size_per_gpu = 4
# dataloader num workers
train_num_workers = 2
log_interval = 1
#####################
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
default_hooks = dict(
    logger=dict(interval=log_interval),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    )
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=max_epochs,  # Maximum training epochs: 300 epochs
    val_interval=save_checkpoint_intervals)  # Validation intervals. Run validation every 10 epochs.

#####################

model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))