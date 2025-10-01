from typing import Literal

import numpy as np
import scipy.ndimage
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

GlobalHydra.instance().clear()


class MaskSegmentor:
    def __init__(self, cfg: DictConfig, device: Literal["cpu", "cuda"] = "cuda"):
        self.device = torch.device(device)

        self.predictor = SAM2ImagePredictor(
            build_sam2(cfg.model_cfg_file, cfg.ckpt_file, device=self.device)
        )

        self.fill_hole = cfg.fill_hole

    @torch.inference_mode()
    def predict(self, color_image: np.ndarray, bboxes: np.ndarray):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            self.predictor.set_image(color_image)
            masks = self.predictor.predict(box=bboxes, multimask_output=False)[0]

        masks = masks.squeeze(1) > 0

        if self.fill_hole:
            for i in range(len(masks)):
                masks[i] = scipy.ndimage.binary_fill_holes(masks[i])

        return masks
