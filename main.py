import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import pyvista as pv
from omegaconf import DictConfig
from PIL import Image

from util import (
    Camera,
    DepthPredictor,
    FlowPredictor,
    GraspEstimator,
    MaskSegmentor,
    ObjectDetector,
    create_gripper,
)


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_pair(
    masks: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, intersection_ratio: float
):
    assert len(masks) == len(labels) and len(bboxes) == len(labels)
    assert np.unique(labels).shape[0] == 2

    # label == 0: holdable mask
    # label == 1: movable mask

    holdable_bboxes = bboxes[labels == 0]
    holdable_masks = masks[labels == 0]
    movable_bboxes = bboxes[labels == 1]
    movable_masks = masks[labels == 1]

    lt = np.maximum(
        holdable_bboxes[:, None, :2], movable_bboxes[None, :, :2]
    )  # [N, M, 2]
    rb = np.minimum(
        holdable_bboxes[:, None, 2:], movable_bboxes[None, :, 2:]
    )  # [N, M, 2]
    wh = (rb - lt).clip(min=0)  # [N, M, 2]
    intersection_ratios = (wh[:, :, 0] * wh[:, :, 1]) / (
        (holdable_bboxes[:, None, 2] - holdable_bboxes[:, None, 0])
        * (holdable_bboxes[:, None, 3] - holdable_bboxes[:, None, 1])
    ).clip(min=1)  # [N, M]

    paired_holdable_bboxes = []
    paired_movable_bboxes = []
    paired_holdable_masks = []
    paired_movable_masks = []

    unpaired_mask = np.ones(len(movable_bboxes), dtype=np.bool_)
    for i in range(intersection_ratios.shape[0]):
        paired_holdable_bboxes.append(holdable_bboxes[i])
        paired_holdable_masks.append(holdable_masks[i])

        j = np.argmax(intersection_ratios[i])

        if intersection_ratios[i, j] < intersection_ratio:
            paired_movable_bboxes.append(None)
            paired_movable_masks.append(None)
        else:
            paired_movable_bboxes.append(movable_bboxes[j])
            paired_movable_masks.append(movable_masks[j])
            unpaired_mask[j] = False

    for j in range(intersection_ratios.shape[1]):
        if unpaired_mask[j]:
            paired_holdable_bboxes.append(None)
            paired_holdable_masks.append(None)
            paired_movable_bboxes.append(movable_bboxes[j])
            paired_movable_masks.append(movable_masks[j])

    return (
        paired_holdable_masks,
        paired_movable_masks,
        paired_holdable_bboxes,
        paired_movable_bboxes,
    )


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera(cfg.data.cam_cfg_file)
    depth_predictor = DepthPredictor(cfg.depth_predictor, device=cfg.device)
    object_detector = ObjectDetector(cfg.object_detector, device=cfg.device)
    mask_segmentor = MaskSegmentor(cfg.mask_segmentor, device=cfg.device)
    grasp_estimator = GraspEstimator(cfg.grasp_estimator)
    flow_predictor = FlowPredictor(cfg.flow_predictor, device=cfg.device)

    color_image = cv2.cvtColor(
        cv2.imread(cfg.data.color_image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    color_image = camera.undistort_image(
        color_image, intrinsics_type="color", interpolation=cv2.INTER_LANCZOS4
    )
    cv2.imwrite(
        (save_dir / "color.png").as_posix(),
        cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR),
    )

    depth_image = (
        np.array(Image.open(cfg.data.depth_image_file)).astype(np.float32) / 1000.0
    )
    depth_image = camera.undistort_image(
        depth_image, intrinsics_type="depth", interpolation=cv2.INTER_NEAREST
    )
    depth_image, valid_mask1 = camera.fitler_depth_by_edge(
        depth_image, rtol=cfg.depth_filter.rtol
    )
    cv2.imwrite(
        (save_dir / "depth.png").as_posix(), (depth_image * 1000).astype(np.uint16)
    )

    normal_image, valid_mask2 = camera.depth_to_normal(
        depth_image, intrinsics_type="depth"
    )
    valid_mask = valid_mask1 & valid_mask2
    normal_image[~valid_mask] = 0
    cv2.imwrite(
        (save_dir / "normal.png").as_posix(),
        cv2.cvtColor(
            ((normal_image + 1) / 2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        ),
    )

    if cfg.verbose:
        save_verbose_dir = save_dir / "verbose"
        save_verbose_dir.mkdir(parents=True, exist_ok=True)

        points_depth, valid_mask1 = camera.unproject(
            depth_image, intrinsics_type="depth"
        )
        points = camera.transform_depth_to_color(points_depth)
        color_image_remapped, valid_mask2 = camera.remap_color_to_depth(
            color_image, depth_image, interpolation=cv2.INTER_LANCZOS4
        )
        valid_mask = valid_mask1 & valid_mask2

        poly_data = pv.PolyData(points[valid_mask].reshape(-1, 3))
        poly_data.point_data["rgb"] = color_image_remapped[valid_mask].reshape(-1, 3)
        poly_data.point_data["normal"] = normal_image[valid_mask].reshape(-1, 3)
        poly_data.save((save_verbose_dir / "pointcloud.vtp").as_posix())

    logging.info("Start depth prediction")

    depth_image_predicted, valid_mask1 = depth_predictor.predict(
        color_image, depth_image, camera
    )
    depth_image_predicted, valid_mask2 = camera.fitler_depth_by_edge(
        depth_image_predicted, rtol=cfg.depth_filter.rtol
    )
    cv2.imwrite(
        (save_dir / "depth_predicted.png").as_posix(),
        (depth_image_predicted * 1000).astype(np.uint16),
    )

    normal_image_predicted, valid_mask3 = camera.depth_to_normal(
        depth_image_predicted, intrinsics_type="color"
    )
    valid_mask = valid_mask1 & valid_mask2 & valid_mask3
    normal_image_predicted[~valid_mask] = 0
    cv2.imwrite(
        (save_dir / "normal_predicted.png").as_posix(),
        cv2.cvtColor(
            ((normal_image_predicted + 1) / 2 * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )

    logging.info("Finish depth prediction")

    if cfg.verbose:
        save_verbose_dir = save_dir / "verbose"
        save_verbose_dir.mkdir(parents=True, exist_ok=True)

        points, valid_mask = camera.unproject(
            depth_image_predicted, intrinsics_type="color"
        )

        poly_data = pv.PolyData(points[valid_mask].reshape(-1, 3))
        poly_data.point_data["rgb"] = color_image[valid_mask].reshape(-1, 3)
        poly_data.point_data["normal"] = normal_image_predicted[valid_mask].reshape(
            -1, 3
        )
        poly_data.save((save_verbose_dir / "pointcloud_predicted.vtp").as_posix())

    logging.info("Start object detection and mask segmentation")

    bboxes, lables, _ = object_detector.predict(color_image)
    assert bboxes is not None, "No object detected"

    masks = mask_segmentor.predict(color_image, bboxes)

    holdable_masks, movable_masks, holdable_bboxes, movable_bboxes = make_pair(
        masks, bboxes, lables, intersection_ratio=cfg.make_pair_intersection_ratio
    )

    with open(save_dir / "holdable_bbox.json", "w") as f:
        json.dump(
            {
                f"{i:05d}": bbox.tolist() if bbox is not None else None
                for i, bbox in enumerate(holdable_bboxes)
            },
            f,
            indent=4,
        )
    with open(save_dir / "movable_bbox.json", "w") as f:
        json.dump(
            {
                f"{i:05d}": bbox.tolist() if bbox is not None else None
                for i, bbox in enumerate(movable_bboxes)
            },
            f,
            indent=4,
        )

    save_holdable_mask_dir = save_dir / "holdable_mask"
    save_holdable_mask_dir.mkdir(parents=True, exist_ok=True)
    save_movable_mask_dir = save_dir / "movable_mask"
    save_movable_mask_dir.mkdir(parents=True, exist_ok=True)
    for i, (holdable_mask, movable_mask) in enumerate(
        zip(holdable_masks, movable_masks)
    ):
        if holdable_mask is not None:
            cv2.imwrite(
                (save_holdable_mask_dir / f"{i:05d}.png").as_posix(),
                (holdable_mask * 255).astype(np.uint8),
            )
        if movable_mask is not None:
            cv2.imwrite(
                (save_movable_mask_dir / f"{i:05d}.png").as_posix(),
                (movable_mask * 255).astype(np.uint8),
            )

    logging.info("Finish object detection and mask segmentation")

    if cfg.verbose:
        save_verbose_dir = save_dir / "verbose"
        save_verbose_dir.mkdir(parents=True, exist_ok=True)

        bbox_viz_image = color_image.copy()
        for holdable_bbox, movable_bbox in zip(holdable_bboxes, movable_bboxes):
            if holdable_bbox is not None:
                x1, y1, x2, y2 = holdable_bbox
                cv2.rectangle(bbox_viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bbox_viz_image,
                    "holdable",
                    (x1, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            if movable_bbox is not None:
                x1, y1, x2, y2 = movable_bbox
                cv2.rectangle(bbox_viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    bbox_viz_image,
                    "movable",
                    (x1, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
        cv2.imwrite(
            (save_verbose_dir / "bbox_viz.png").as_posix(),
            cv2.cvtColor(bbox_viz_image, cv2.COLOR_RGB2BGR),
        )

    logging.info("Start grasp estimation")

    grasp_infos = grasp_estimator.predict(
        depth_image_predicted,
        normal_image_predicted,
        holdable_masks,
        movable_masks,
        camera,
    )

    with open(save_dir / "grasp_info.json", "w") as f:
        json.dump(
            {
                key: {
                    "pose": value["pose"].tolist(),
                    "width": float(value["width"]),
                    "depth": float(value["depth"]),
                }
                if value is not None
                else None
                for key, value in grasp_infos.items()
            },
            f,
            indent=4,
        )

    logging.info("Finish grasp estimation")

    if cfg.verbose:
        save_verbose_dir = save_dir / "verbose"
        save_verbose_dir.mkdir(parents=True, exist_ok=True)

        for name, grasp_info in grasp_infos.items():
            if grasp_info is None:
                continue

            pose = grasp_info["pose"]
            width = grasp_info["width"]
            depth = grasp_info["depth"]
            gripper_mesh = create_gripper(
                center=pose[:3, 3],
                R=pose[:3, :3],
                width=width,
                depth=depth,
                height=0.02,
                finger_width=0.004,
            )
            gripper_mesh.export(save_verbose_dir / f"gripper_{name}.ply")

    logging.info("Start flow prediction")

    directions = flow_predictor.predict(
        depth_image_predicted,
        normal_image_predicted,
        movable_masks,
        grasp_infos,
        camera,
    )

    with open(save_dir / "direction.json", "w") as f:
        json.dump(
            {
                key: value.tolist() if value is not None else None
                for key, value in directions.items()
            },
            f,
            indent=4,
        )

    logging.info("Finish flow prediction")

    if cfg.verbose:
        save_verbose_dir = save_dir / "verbose"
        save_verbose_dir.mkdir(parents=True, exist_ok=True)

        for name, direction in directions.items():
            if direction is None:
                continue

            direction_arrow = pv.Arrow(
                start=grasp_infos[name]["pose"][:3, 3],
                direction=direction,
                tip_resolution=64,
                shaft_resolution=64,
                scale=0.15,
            )
            direction_arrow.save(save_verbose_dir / f"direction_{name}.ply")


if __name__ == "__main__":
    main()
