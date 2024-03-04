from mmdet.registry import VISUALIZERS
from mmdet.registry import DATASETS
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmdet.utils import register_all_modules
#from mmengine.visualization import Visualizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def registerStuff():
    register_all_modules()
    @DATASETS.register_module()
    class MoNuSegDataset(YOLOv5CocoDataset):
        """MoNuSeg dataset."""
        METAINFO = {
            'classes': ("nucleus"),
            'palette':[
                    (0, 255, 0),
            ],  # This has been changed, lets see what it does
        }
        def __init__(
                self,
                #img_suffix=".png",
                seg_map_suffix=".png",
                return_classes=False,
                **kwargs,
            ) -> None:
                self.return_classes = return_classes
                super().__init__(
                    #img_suffix=img_suffix,
                    seg_map_suffix=seg_map_suffix,
                    **kwargs,
                )