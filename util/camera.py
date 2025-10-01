from configparser import ConfigParser
from typing import Literal

import cv2
import numpy as np
from utils3d.numpy.utils import depth_edge


class Camera:
    def __init__(self, cfg_file: str):
        self._parse_cam_cfg(cfg_file)

    def _parse_cam_cfg(self, cfg_file: str):
        cfg_parser = ConfigParser()
        assert cfg_parser.read(cfg_file) == [cfg_file]

        color_intrinsics_distorted = np.array(
            [
                [
                    float(cfg_parser["ColorIntrinsic"]["fx"]),
                    0,
                    float(cfg_parser["ColorIntrinsic"]["cx"]),
                ],
                [
                    0,
                    float(cfg_parser["ColorIntrinsic"]["fy"]),
                    float(cfg_parser["ColorIntrinsic"]["cy"]),
                ],
                [0, 0, 1],
            ]
        )

        color_distortions = np.array(
            [
                float(cfg_parser["ColorDistortion"]["k1"]),
                float(cfg_parser["ColorDistortion"]["k2"]),
                float(cfg_parser["ColorDistortion"]["p1"]),
                float(cfg_parser["ColorDistortion"]["p2"]),
                float(cfg_parser["ColorDistortion"]["k3"]),
                float(cfg_parser["ColorDistortion"].get("k4", 0)),
                float(cfg_parser["ColorDistortion"].get("k5", 0)),
                float(cfg_parser["ColorDistortion"].get("k6", 0)),
            ]
        )

        color_image_size = (
            int(cfg_parser["ColorIntrinsic"]["width"]),
            int(cfg_parser["ColorIntrinsic"]["height"]),
        )

        color_intrinsics, color_mapx, color_mapy = self._undistort_intrinsics(
            color_intrinsics_distorted, color_distortions, color_image_size
        )

        depth_intrinsics_distorted = np.array(
            [
                [
                    float(cfg_parser["DepthIntrinsic"]["fx"]),
                    0,
                    float(cfg_parser["DepthIntrinsic"]["cx"]),
                ],
                [
                    0,
                    float(cfg_parser["DepthIntrinsic"]["fy"]),
                    float(cfg_parser["DepthIntrinsic"]["cy"]),
                ],
                [0, 0, 1],
            ]
        )

        depth_distortions = np.array(
            [
                float(cfg_parser["DepthDistortion"]["k1"]),
                float(cfg_parser["DepthDistortion"]["k2"]),
                float(cfg_parser["DepthDistortion"]["p1"]),
                float(cfg_parser["DepthDistortion"]["p2"]),
                float(cfg_parser["DepthDistortion"]["k3"]),
                float(cfg_parser["DepthDistortion"].get("k4", 0)),
                float(cfg_parser["DepthDistortion"].get("k5", 0)),
                float(cfg_parser["DepthDistortion"].get("k6", 0)),
            ]
        )

        depth_image_size = (
            int(cfg_parser["DepthIntrinsic"]["width"]),
            int(cfg_parser["DepthIntrinsic"]["height"]),
        )

        depth_intrinsics, depth_mapx, depth_mapy = self._undistort_intrinsics(
            depth_intrinsics_distorted, depth_distortions, depth_image_size
        )

        tf_depth_to_color = np.array(
            [
                [
                    float(cfg_parser["D2CTransformParam"]["rot0"]),
                    float(cfg_parser["D2CTransformParam"]["rot1"]),
                    float(cfg_parser["D2CTransformParam"]["rot2"]),
                    float(cfg_parser["D2CTransformParam"]["trans0"]),
                ],
                [
                    float(cfg_parser["D2CTransformParam"]["rot3"]),
                    float(cfg_parser["D2CTransformParam"]["rot4"]),
                    float(cfg_parser["D2CTransformParam"]["rot5"]),
                    float(cfg_parser["D2CTransformParam"]["trans1"]),
                ],
                [
                    float(cfg_parser["D2CTransformParam"]["rot6"]),
                    float(cfg_parser["D2CTransformParam"]["rot7"]),
                    float(cfg_parser["D2CTransformParam"]["rot8"]),
                    float(cfg_parser["D2CTransformParam"]["trans2"]),
                ],
                [0, 0, 0, 1],
            ],
        )
        tf_depth_to_color[:3, 3] /= 1000.0  # mm to m

        self.color_image_size = color_image_size
        self.color_intrinsics = color_intrinsics
        self.color_mapx = color_mapx
        self.color_mapy = color_mapy

        self.depth_image_size = depth_image_size
        self.depth_intrinsics = depth_intrinsics
        self.depth_mapx = depth_mapx
        self.depth_mapy = depth_mapy

        self.tf_depth_to_color = tf_depth_to_color

    def _undistort_intrinsics(
        self,
        intrinsics_distorted: np.ndarray,
        distortions: np.ndarray,
        image_size: tuple[int, int],
    ):
        intrinsics, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics_distorted,
            distortions,
            image_size,
            0,
            image_size,
        )

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsics_distorted,
            distortions,
            None,
            intrinsics,
            image_size,
            cv2.CV_32FC1,
        )

        return intrinsics, mapx, mapy

    def undistort_image(
        self,
        image: np.ndarray,
        intrinsics_type: Literal["color", "depth"],
        interpolation=cv2.INTER_NEAREST,
    ):
        assert intrinsics_type in ["color", "depth"]

        return cv2.remap(
            image,
            self.color_mapx if intrinsics_type == "color" else self.depth_mapx,
            self.color_mapy if intrinsics_type == "color" else self.depth_mapy,
            interpolation=interpolation,
        )

    def unproject(
        self, depth_image: np.ndarray, intrinsics_type: Literal["color", "depth"]
    ):
        assert intrinsics_type in ["color", "depth"]

        u, v = np.meshgrid(
            np.arange(depth_image.shape[1]),
            np.arange(depth_image.shape[0]),
            indexing="xy",
        )

        if intrinsics_type == "color":
            fx, fy = self.color_intrinsics[0, 0], self.color_intrinsics[1, 1]
            cx, cy = self.color_intrinsics[0, 2], self.color_intrinsics[1, 2]
        else:
            fx, fy = self.depth_intrinsics[0, 0], self.depth_intrinsics[1, 1]
            cx, cy = self.depth_intrinsics[0, 2], self.depth_intrinsics[1, 2]

        z = depth_image.astype(np.float32)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        points = np.stack([x, y, z], axis=-1)

        valid_mask = z > 0

        return points, valid_mask

    def transform_depth_to_color(self, points_depth: np.ndarray):
        if points_depth.ndim == 2:
            points_color = points_depth @ self.tf_depth_to_color[
                :3, :3
            ].T + self.tf_depth_to_color[:3, 3].reshape(1, 3)
        elif points_depth.ndim == 3:
            points_color = points_depth @ self.tf_depth_to_color[
                :3, :3
            ].T + self.tf_depth_to_color[:3, 3].reshape(1, 1, 3)
        else:
            raise ValueError("points_depth.ndim should be 2 or 3")

        return points_color

    def remap_color_to_depth(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        interpolation=cv2.INTER_NEAREST,
    ):
        points_depth, valid_mask = self.unproject(depth_image, intrinsics_type="depth")

        points_color = self.transform_depth_to_color(points_depth)

        u = (
            points_color[..., 0] / points_color[..., 2] * self.color_intrinsics[0, 0]
            + self.color_intrinsics[0, 2]
        )
        v = (
            points_color[..., 1] / points_color[..., 2] * self.color_intrinsics[1, 1]
            + self.color_intrinsics[1, 2]
        )

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        valid_mask = valid_mask & (
            (u_int >= 0)
            & (u_int < self.color_image_size[0])
            & (v_int >= 0)
            & (v_int < self.color_image_size[1])
        )

        color_image_remapped = cv2.remap(
            color_image,
            u.astype(np.float32),
            v.astype(np.float32),
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return color_image_remapped, valid_mask

    def depth_to_normal(
        self, depth_image: np.ndarray, intrinsics_type: Literal["color", "depth"]
    ):
        pointcloud_img, valid_mask = self.unproject(
            depth_image, intrinsics_type=intrinsics_type
        )

        dzdy, dzdx = np.gradient(pointcloud_img, axis=(0, 1))

        normal = -np.linalg.cross(
            dzdx.reshape(-1, 3), dzdy.reshape(-1, 3), axis=-1
        ).reshape(pointcloud_img.shape)
        normal /= np.linalg.norm(normal, ord=2, axis=-1, keepdims=True).clip(min=1e-15)

        valid_mask = valid_mask & (np.linalg.norm(normal, ord=2, axis=-1) > 0)

        normal[~valid_mask] = 0

        return normal, valid_mask

    def fitler_depth_by_edge(
        self,
        depth_image: np.ndarray,
        atol: float = None,
        rtol: float = None,
        kernel_size: int = 3,
    ):
        invalid_mask = depth_edge(
            depth_image, atol=atol, rtol=rtol, kernel_size=kernel_size
        )
        depth_image[invalid_mask] = 0

        return depth_image, ~invalid_mask
