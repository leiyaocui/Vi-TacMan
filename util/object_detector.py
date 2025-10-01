import logging
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from torchvision.ops import batched_nms
from torchvision.transforms import Compose


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: np.ndarray):
        image = TF.resize(
            image,
            [self.size, self.size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        return image


class ToTensor:
    def __call__(self, image: np.ndarray):
        return TF.to_tensor(image)


class Normalize:
    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor):
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image


class ObjectDetector:
    def __init__(self, cfg: DictConfig, device: Literal["cpu", "cuda"] = "cuda"):
        self.device = torch.device(device)

        self.model = torch.hub.load(
            cfg.dinov3_dir,
            cfg.dinov3_type,
            num_classes=2,
            source="local",
            pretrained=False,
        )
        self.model = self.model.to(self.device).eval()

        dinov3_ckpt = torch.load(
            cfg.dinov3_ckpt_file, map_location=self.device, weights_only=True
        )
        self.model.detector.backbone[0]._backbone.backbone.load_state_dict(dinov3_ckpt)

        detr_ckpt = torch.load(
            cfg.detr_ckpt_file, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(detr_ckpt, strict=False)

        self.transform = Compose(
            [
                ToTensor(),
                Resize(1024),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.score_threshold = cfg.score_threshold
        self.nms_threshold = cfg.nms_threshold

    @torch.inference_mode()
    def predict(self, color_image: np.ndarray):
        inputs = self.transform(color_image)
        inputs = [inputs.to(self.device)]

        outputs = self.model(inputs)[0]

        scores = outputs["scores"]
        labels = outputs["labels"]
        bboxes = outputs["boxes"]  # (x0, y0, x1, y1)

        valid_mask = scores >= self.score_threshold

        scores = scores[valid_mask]
        labels = labels[valid_mask]
        bboxes = bboxes[valid_mask]

        if bboxes.shape[0] == 0:
            logging.warning("No objects detected")

            return None, None, None

        valid_mask = batched_nms(bboxes, scores, labels, self.nms_threshold)

        scores = scores[valid_mask].cpu().numpy()
        labels = labels[valid_mask].cpu().numpy()
        bboxes = bboxes[valid_mask].cpu().numpy()

        bboxes[:, 0] *= color_image.shape[1] / 1024
        bboxes[:, 1] *= color_image.shape[0] / 1024
        bboxes[:, 2] *= color_image.shape[1] / 1024
        bboxes[:, 3] *= color_image.shape[0] / 1024

        bboxes = np.round(bboxes).astype(np.int32)
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, color_image.shape[1] - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, color_image.shape[0] - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, color_image.shape[1] - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, color_image.shape[0] - 1)

        return bboxes, labels, scores
