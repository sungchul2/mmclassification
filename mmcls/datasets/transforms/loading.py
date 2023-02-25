from typing import Optional

import numpy as np
from mmcv import TRANSFORMS
from mmcv.transforms import LoadImageFromFile
from PIL import Image


@TRANSFORMS.register_module()
class AIELoadImageFromFile(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_path']
        img = np.array(Image.open(filename).convert("RGB"))
        if self.to_float32:
            img = img.astype(np.float32)

        if img.shape[2] == 4:
            img = img[...,:-1]

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
