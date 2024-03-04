from mmyolo.registry import VISUALIZERS
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmdet.utils import register_all_modules
from mmengine.visualization import Visualizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def registerStuff():
    register_all_modules()
    @VISUALIZERS.register_module()
    class DetLocalVisualizer(Visualizer):
        def __init__(self,
                    name: str = 'visualizer',
                    image: Optional[np.ndarray] = None,
                    vis_backends: Optional[Dict] = None,
                    save_dir: Optional[str] = None,
                    bbox_color: Optional[Union[str, Tuple[int]]] = None,
                    text_color: Optional[Union[str,
                                                Tuple[int]]] = (200, 200, 200),
                    mask_color: Optional[Union[str, Tuple[int]]] = None,
                    line_width: Union[int, float] = 3,
                    alpha: float = 0.8) -> None:
            super().__init__(
                name=name,
                image=image,
                vis_backends=vis_backends,
                save_dir=save_dir)
            self.bbox_color = bbox_color
            self.text_color = text_color
            self.mask_color = mask_color
            self.line_width = line_width
            self.alpha = alpha
            # Set default value. When calling
            # `DetLocalVisualizer().dataset_meta=xxx`,
            # it will override the default value.
            self.dataset_meta = {}
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
                seg_map_suffix=".png",
                return_classes=False,
                **kwargs,
            ) -> None:
                self.return_classes = return_classes
                super().__init__(
                    seg_map_suffix=seg_map_suffix,
                    **kwargs,
                )