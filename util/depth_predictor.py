from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.linear_model import RANSACRegressor
from torchvision.transforms import Compose

from thirdparty.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from thirdparty.depth_anything_v2.depth_anything_v2.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)

from .camera import Camera


class DepthPredictor:
    def __init__(self, cfg: DictConfig, device: Literal["cpu", "cuda"] = "cuda"):
        self.device = torch.device(device)

        self.model = DepthAnythingV2(
            encoder=cfg.encoder, features=cfg.features, out_channels=cfg.out_channels
        )
        self.model = self.model.to(self.device).eval()

        ckpt = torch.load(cfg.ckpt_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)

        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    @torch.inference_mode()
    def predict(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        camera: Camera,
        depth_range: tuple[float, float] | None = None,
    ):
        inputs = self.transform({"image": color_image / 255.0})["image"]
        inputs = torch.from_numpy(inputs).unsqueeze(0).to(self.device)

        outputs = self.model.forward(inputs)

        disparity_image_predcited = outputs
        disparity_image_predcited = (
            F.interpolate(
                disparity_image_predcited.unsqueeze(1),
                (color_image.shape[0], color_image.shape[1]),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            .cpu()
            .numpy()
        )

        disparity_image_predcited_remapped, valid_mask = camera.remap_color_to_depth(
            disparity_image_predcited, depth_image, interpolation=cv2.INTER_NEAREST
        )

        with np.errstate(divide="ignore"):
            disparity_image = 1.0 / depth_image

        valid_mask = valid_mask & np.isfinite(disparity_image)
        if depth_range is not None:
            valid_mask = (
                valid_mask
                & (depth_image > depth_range[0])
                & (depth_image < depth_range[1])
            )

        ransac_regressor = RANSACRegressor(random_state=0)
        ransac_regressor.fit(
            disparity_image_predcited_remapped[valid_mask].reshape(-1, 1),
            disparity_image[valid_mask].reshape(-1, 1),
        )

        disparity_image_predcited = ransac_regressor.predict(
            disparity_image_predcited.reshape(-1, 1)
        ).reshape(color_image.shape[0], color_image.shape[1])

        depth_image_predicted = 1.0 / disparity_image_predcited
        depth_image_predicted = np.nan_to_num(
            depth_image_predicted, nan=0.0, posinf=0.0, neginf=0.0
        )

        valid_mask = disparity_image_predcited > 0

        depth_image_predicted[~valid_mask] = 0

        return depth_image_predicted, valid_mask
