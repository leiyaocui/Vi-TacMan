import logging
from typing import Literal

import cv2
import numpy as np
import trimesh
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from .camera import Camera


def create_box(
    width: float,
    height: float,
    depth: float,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
):
    vertices = np.array(
        [
            [0, 0, 0],
            [width, 0, 0],
            [0, 0, depth],
            [width, 0, depth],
            [0, height, 0],
            [width, height, 0],
            [0, height, depth],
            [width, height, depth],
        ]
    )
    vertices[:, 0] += dx
    vertices[:, 1] += dy
    vertices[:, 2] += dz

    faces = np.array(
        [
            [4, 7, 5],
            [4, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
        ]
    )

    box = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return box


def create_gripper(
    center: np.ndarray,
    R: np.ndarray,
    width: float,
    depth: float,
    height: float,
    finger_width: float,
) -> trimesh.Trimesh:
    """
    Parameters:
        center: numpy array of (3,), target point as gripper center
        R: numpy array of (3,3), rotation matrix of gripper
        width: float, gripper width
        depth: float, gripper depth
        height: float, gripper height
        finger_width: float, gripper finger width

    Returns:
        trimesh.Trimesh
    """

    tail_length = 0.04
    depth_base = 0.02

    left = create_box(depth + depth_base + finger_width, finger_width, height)
    right = create_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_box(finger_width, width, height)
    tail = create_box(tail_length, finger_width, height)

    left.apply_translation(
        [-depth_base - finger_width, -width / 2 - finger_width, -height / 2]
    )
    right.apply_translation([-depth_base - finger_width, width / 2, -height / 2])
    bottom.apply_translation([-finger_width - depth_base, -width / 2, -height / 2])
    tail.apply_translation(
        [-tail_length - finger_width - depth_base, -finger_width / 2, -height / 2]
    )

    gripper = trimesh.util.concatenate([left, right, bottom, tail])
    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = center
    gripper.apply_transform(tf)

    return gripper


def detect_gripper_finger_collision(
    points: np.ndarray,
    center: np.ndarray,
    R: np.ndarray,
    width: float,
    depth: float,
    height: float,
    finger_width: float,
):
    depth_base = 0.02

    left = create_box(depth + depth_base + finger_width, finger_width, height)
    right = create_box(depth + depth_base + finger_width, finger_width, height)

    left.apply_translation(
        [-depth_base - finger_width, -width / 2 - finger_width, -height / 2]
    )
    right.apply_translation([-depth_base - finger_width, width / 2, -height / 2])

    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = center
    left.apply_transform(tf)
    right.apply_transform(tf)

    left_hull = left.convex_hull
    right_hull = right.convex_hull

    is_collided = left_hull.contains(points).any() or right_hull.contains(points).any()

    return is_collided


class GraspEstimator:
    def __init__(self, cfg: DictConfig):
        self.depth = cfg.depth
        self.height = cfg.height
        self.finger_width = cfg.finger_width

        self.theta_step = cfg.theta_step

        self.min_width = cfg.min_width
        self.max_width = cfg.max_width
        self.width_tol = cfg.width_tol

        self.min_approaching_depth = cfg.min_approaching_depth
        self.max_approaching_depth = cfg.max_approaching_depth
        self.approaching_depth_tol = cfg.approaching_depth_tol

        assert self.max_approaching_depth <= self.depth

    def predict(
        self,
        depth_image: np.ndarray,
        normal_image: np.ndarray,
        holdable_masks: list[np.ndarray],
        movable_masks: list[np.ndarray],
        camera: Camera,
    ):
        pointcloud_image, valid_mask = camera.unproject(
            depth_image, intrinsics_type="color"
        )

        grasp_infos = {}
        for i, (holdable_mask, movable_mask) in enumerate(
            zip(holdable_masks, movable_masks)
        ):
            if holdable_mask is None or movable_mask is None:
                grasp_infos[f"{i:05d}"] = None
                continue

            pose, width, depth = self._predict_single(
                pointcloud_image, normal_image, valid_mask, holdable_mask, movable_mask
            )
            if pose is None:
                grasp_infos[f"{i:05d}"] = None
            else:
                grasp_infos[f"{i:05d}"] = {
                    "pose": pose,
                    "width": width,
                    "depth": depth,
                }

        return grasp_infos

    def _predict_single(
        self,
        pointcloud_image: np.ndarray,
        normal_image: np.ndarray,
        valid_mask: np.ndarray,
        holdable_mask: np.ndarray,
        movable_mask: np.ndarray,
    ):
        normal_mask1 = movable_mask & (~holdable_mask) & valid_mask
        normal_mask1 = (
            cv2.erode(normal_mask1.astype(np.uint8), np.ones((3, 3)), iterations=3) > 0
        )

        approaching_dir = -normal_image[normal_mask1].mean(axis=0)
        if np.linalg.norm(approaching_dir) == 0:
            logging.warning("Invalid approaching direction")

            return None, None, None

        approaching_dir /= np.linalg.norm(
            approaching_dir, ord=2, axis=-1, keepdims=True
        ).clip(min=1e-15)

        rot_gripper_to_world = self._approaching_dir_to_rot(approaching_dir)

        holdable_points = pointcloud_image[holdable_mask & valid_mask]
        if holdable_points.shape[0] == 0:
            logging.warning("No holdable points")

            return None, None, None

        movable_points = pointcloud_image[movable_mask & (~holdable_mask) & valid_mask]
        if movable_points.shape[0] == 0:
            logging.warning("No movable points")

            return None, None, None

        normal_mask2 = holdable_mask & valid_mask
        holdable_point_weights = np.sum(
            normal_image[normal_mask2] * -approaching_dir.reshape(1, 3), axis=-1
        ).clip(0.0, 1.0)
        if holdable_point_weights.sum() == 0:
            logging.warning("Invalid holdable_point_weights")

            return None, None, None

        grasping_point = (holdable_point_weights.reshape(-1, 1) * holdable_points).sum(
            axis=0
        ) / holdable_point_weights.sum()

        pose, width, depth = self._search_optimal_grasp(
            rot_gripper_to_world, grasping_point, holdable_points, movable_points
        )

        if pose is None:
            logging.warning("No valid grasp found")

            return None, None, None

        return pose, width, depth

    def _approaching_dir_to_rot(self, approaching_dir: np.ndarray):
        x_axis = approaching_dir
        x_axis /= np.linalg.norm(x_axis, ord=2, axis=-1, keepdims=True).clip(min=1e-15)

        y_axis = np.linalg.cross(x_axis, np.array([0.0, 0.0, 1.0]), axis=0)
        if np.linalg.norm(y_axis) < 1e-15:
            y_axis = np.linalg.cross(x_axis, np.array([0.0, 1.0, 0.0]), axis=0)
        y_axis /= np.linalg.norm(y_axis, ord=2, axis=-1, keepdims=True).clip(min=1e-15)

        z_axis = np.linalg.cross(x_axis, y_axis, axis=0)
        z_axis /= np.linalg.norm(z_axis, ord=2, axis=-1, keepdims=True).clip(min=1e-15)

        rot_gripper_to_world = np.stack([x_axis, y_axis, z_axis], axis=-1)

        return rot_gripper_to_world

    def _search_optimal_grasp(
        self,
        rot_gripper_to_world: np.ndarray,
        grasping_point: np.ndarray,
        holdable_points: np.ndarray,
        movable_points: np.ndarray,
    ):
        thetas = np.arange(0.0, np.pi, np.deg2rad(self.theta_step))

        rot_grasp_to_world = (
            rot_gripper_to_world[None, ...]
            @ Rotation.from_euler("X", thetas, degrees=False).as_matrix()
        )

        valid_mask = np.ones(thetas.shape[0], dtype=np.bool)

        low_width, high_width = self.min_width, self.max_width
        while (high_width - low_width) > self.width_tol:
            mid_width = (low_width + high_width) / 2.0

            valid_mask_curr = valid_mask.copy()
            for i in range(thetas.shape[0]):
                if not valid_mask_curr[i]:
                    continue

                is_collided = detect_gripper_finger_collision(
                    holdable_points,
                    grasping_point,
                    rot_grasp_to_world[i],
                    mid_width,
                    self.depth,
                    self.height,
                    self.finger_width,
                )
                valid_mask_curr[i] = not is_collided

            if np.any(valid_mask_curr):
                high_width = mid_width
                valid_mask = valid_mask_curr
            else:
                low_width = mid_width

        width = mid_width
        rot_grasp_to_world = rot_grasp_to_world[valid_mask][0]

        low_approaching_depth, high_approaching_depth = (
            self.min_approaching_depth,
            self.max_approaching_depth,
        )
        while (
            high_approaching_depth - low_approaching_depth
        ) > self.approaching_depth_tol:
            mid_approaching_depth = (
                low_approaching_depth + high_approaching_depth
            ) / 2.0

            is_collided = detect_gripper_finger_collision(
                movable_points,
                grasping_point - rot_grasp_to_world[:, 0] * mid_approaching_depth,
                rot_grasp_to_world,
                width,
                self.depth,
                self.height,
                self.finger_width,
            )
            if is_collided:
                low_approaching_depth = mid_approaching_depth
            else:
                high_approaching_depth = mid_approaching_depth

        approaching_depth = mid_approaching_depth

        pose = np.eye(4)
        pose[:3, :3] = rot_grasp_to_world
        pose[:3, 3] = grasping_point - rot_grasp_to_world[:, 0] * approaching_depth

        return pose, width, self.depth
