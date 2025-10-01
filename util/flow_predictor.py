import logging
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch3d.ops import (
    ball_query,
    corresponding_points_alignment,
    knn_gather,
    knn_points,
    sample_farthest_points,
)
from pytorch3d.ops.utils import masked_gather

from .camera import Camera
from .vmf_distribution import SteinVMFDistribution


class SetAbstractionMSG(nn.Module):
    def __init__(
        self,
        num_centroids: int,
        radius_list: list[float],
        num_samples_list: list[int],
        in_channel: int,
        out_channel_list: list[list[int]],
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.radius_list = radius_list
        self.num_samples_list = num_samples_list

        self.mlps = nn.ModuleList()
        for i in range(len(out_channel_list)):
            mlp = nn.Sequential()
            last_channel = in_channel + 3
            for out_channel in out_channel_list[i]:
                mlp.append(
                    nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False)
                )
                mlp.append(nn.BatchNorm2d(out_channel))
                mlp.append(nn.ReLU())
                last_channel = out_channel
            self.mlps.append(mlp)

    def forward(self, point: torch.Tensor, point_feat: torch.Tensor = None):
        # point: (B, N, 3), point_feat: (B, N, D)

        # centroid: (B, S, 3)
        centroid, _ = sample_farthest_points(
            point, K=self.num_centroids, random_start_point=False
        )

        point_feat_new_list = []
        for i, (radius, num_samples) in enumerate(
            zip(self.radius_list, self.num_samples_list)
        ):
            # nn_idx: (B, S, K), point_grouped: (B, S, K, 3)
            _, nn_idx, point_grouped = ball_query(
                centroid, point, radius=radius, K=num_samples, return_nn=True
            )
            point_grouped -= centroid.unsqueeze(-2)

            # point_feat_grouped: (B, S, K, D)
            if point_feat is not None:
                point_feat_grouped = masked_gather(point_feat, nn_idx)
                point_feat_grouped = torch.cat(
                    [point_feat_grouped, point_grouped], dim=3
                )
            else:
                point_feat_grouped = point_grouped
            # point_feat_grouped: (B, D, K, S)
            point_feat_grouped = point_feat_grouped.permute(0, 3, 2, 1)
            point_feat_grouped = self.mlps[i](point_feat_grouped)

            # point_feat_new: (B, D, S)
            point_feat_new = torch.max(point_feat_grouped, dim=2)[0]
            point_feat_new_list.append(point_feat_new)

        # point_feat_new: (B, D, S)
        point_feat_new = torch.cat(point_feat_new_list, dim=1)
        # point_feat_new: (B, S, D)
        point_feat_new = point_feat_new.permute(0, 2, 1)

        return centroid, point_feat_new


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, out_channel_list: list[int]):
        super().__init__()

        self.mlps = nn.Sequential()
        last_channel = in_channel
        for out_channel in out_channel_list:
            self.mlps.extend(
                nn.Sequential(
                    nn.Conv1d(last_channel, out_channel, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU(),
                )
            )
            last_channel = out_channel

    def forward(
        self,
        point_1: torch.Tensor,
        point_2: torch.Tensor,
        point_feat_1: torch.Tensor = None,
        point_feat_2: torch.Tensor = None,
    ):
        # point_1: (B, N, 3), point_2: (B, S, 3)
        # point_feat_1: (B, N, D), point_feat_2: (B, S, D)
        N, S = point_1.shape[1], point_2.shape[1]

        if S == 1:
            point_feat_2_interp = point_feat_2.repeat(1, N, 1)
        else:
            # dist: (B, N, K), nn_idx: (B, N, K)
            dist, nn_idx, _ = knn_points(
                point_1, point_2, norm=2, K=3, return_nn=False, return_sorted=True
            )

            dist_recip = 1.0 / (dist + 1e-8)
            # weight: (B, N, K)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            point_feat_2_interp = torch.sum(
                knn_gather(point_feat_2, nn_idx) * weight.unsqueeze(-1), dim=2
            )

        # point_feat_new: (B, N, D)
        if point_feat_1 is not None:
            point_feat_new = torch.cat([point_feat_1, point_feat_2_interp], dim=2)
        else:
            point_feat_new = point_feat_2_interp

        # point_feat_new: (B, D, N)
        point_feat_new = point_feat_new.permute(0, 2, 1)
        point_feat_new = self.mlps(point_feat_new)
        # point_feat_new: (B, N, D)
        point_feat_new = point_feat_new.permute(0, 2, 1)

        return point_feat_new


class PartFlowNet(nn.Module):
    def __init__(self, num_points: int, in_channel: int = 0):
        super().__init__()
        assert num_points % 4 == 0, "num_points must be divisible by 4"
        assert num_points >= 256, "num_points must be at least 256"
        assert in_channel == 4, "in_channel must be 4 (1 for mask and 3 for normals)"

        self.sa1 = SetAbstractionMSG(
            num_centroids=num_points // 4,
            radius_list=[0.05, 0.1],
            num_samples_list=[16, 32],
            in_channel=in_channel,
            out_channel_list=[[16, 16, 32], [32, 32, 64]],
        )
        self.sa2 = SetAbstractionMSG(
            num_centroids=num_points // 4**2,
            radius_list=[0.1, 0.2],
            num_samples_list=[16, 32],
            in_channel=32 + 64,
            out_channel_list=[[64, 64, 128], [64, 96, 128]],
        )
        self.sa3 = SetAbstractionMSG(
            num_centroids=num_points // 4**3,
            radius_list=[0.2, 0.4],
            num_samples_list=[16, 32],
            in_channel=128 + 128,
            out_channel_list=[[128, 196, 256], [128, 196, 256]],
        )
        self.sa4 = SetAbstractionMSG(
            num_centroids=num_points // 4**4,
            radius_list=[0.4, 0.8],
            num_samples_list=[16, 32],
            in_channel=256 + 256,
            out_channel_list=[[256, 256, 512], [256, 384, 512]],
        )

        self.fp4 = FeaturePropagation(
            in_channel=512 + 512 + 256 + 256, out_channel_list=[256, 256]
        )
        self.fp3 = FeaturePropagation(
            in_channel=128 + 128 + 256, out_channel_list=[256, 256]
        )
        self.fp2 = FeaturePropagation(
            in_channel=32 + 64 + 256, out_channel_list=[256, 128]
        )
        self.fp1 = FeaturePropagation(
            in_channel=128 + in_channel, out_channel_list=[128, 128, 128]
        )

        self.mlp_out = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, kernel_size=1, bias=False),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor, normals: torch.Tensor):
        l0_point = points  # (B, N, 3)

        l0_point_feat = torch.cat(
            [
                mask,  # (B, N, 1)
                normals,  # (B, N, 3)
            ],
            dim=-1,
        )  # (B, N, 4)

        l1_point, l1_point_feat = self.sa1(l0_point, l0_point_feat)
        l2_point, l2_point_feat = self.sa2(l1_point, l1_point_feat)
        l3_point, l3_point_feat = self.sa3(l2_point, l2_point_feat)
        l4_point, l4_point_feat = self.sa4(l3_point, l3_point_feat)

        l3_point_feat = self.fp4(l3_point, l4_point, l3_point_feat, l4_point_feat)
        l2_point_feat = self.fp3(l2_point, l3_point, l2_point_feat, l3_point_feat)
        l1_point_feat = self.fp2(l1_point, l2_point, l1_point_feat, l2_point_feat)
        l0_point_feat = self.fp1(l0_point, l1_point, l0_point_feat, l1_point_feat)

        l0_point_feat = l0_point_feat.permute(0, 2, 1)
        out = self.mlp_out(l0_point_feat)
        out = out.permute(0, 2, 1)

        return out


class FlowPredictor:
    def __init__(self, cfg: DictConfig, device: Literal["cpu", "cuda"] = "cuda"):
        self.device = torch.device(device)

        self.model = PartFlowNet(cfg.num_points, in_channel=cfg.in_channel)
        self.model = self.model.to(self.device).eval()

        ckpt = torch.load(cfg.ckpt_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])

        self.downsample_size = cfg.downsample_size
        self.num_neighbors = cfg.num_neighbors

    @torch.inference_mode()
    def predict(
        self,
        depth_image: np.ndarray,
        normal_image: np.ndarray,
        movable_masks: list[np.ndarray],
        grasp_infos: dict[str, dict[str, np.ndarray]],
        camera: Camera,
    ):
        pointcloud_image, valid_mask = camera.unproject(
            depth_image, intrinsics_type="color"
        )

        directions = {}
        for i, movable_mask in enumerate(movable_masks):
            grasp_info = grasp_infos[f"{i:05d}"]

            if movable_mask is None or grasp_info is None:
                directions[f"{i:05d}"] = None
                continue

            grasping_point = grasp_info["pose"][:3, 3]

            direction = self._predict_single(
                pointcloud_image,
                normal_image,
                valid_mask,
                movable_mask,
                grasping_point,
            )
            directions[f"{i:05d}"] = direction

        return directions

    @torch.inference_mode()
    def _predict_single(
        self,
        pointcloud_image: np.ndarray,
        normal_image: np.ndarray,
        valid_mask: np.ndarray,
        movable_mask: np.ndarray,
        grasping_point: np.ndarray,
    ):
        points = torch.tensor(pointcloud_image[valid_mask]).to(
            device=self.device, dtype=torch.float32
        )
        normals = torch.tensor(normal_image[valid_mask]).to(
            device=self.device, dtype=torch.float32
        )
        movable_mask = torch.tensor(movable_mask[valid_mask]).to(
            device=self.device, dtype=torch.float32
        )
        grasping_point = torch.tensor(grasping_point).to(
            device=self.device, dtype=torch.float32
        )

        point_center = points.mean(dim=0)
        points -= point_center.reshape(1, 3)
        grasping_point -= point_center

        if points.shape[0] > self.downsample_size:
            sampled_indices = np.sort(
                np.random.choice(points.shape[0], self.downsample_size, replace=False)
            )

            points = points[sampled_indices]
            normals = normals[sampled_indices]
            movable_mask = movable_mask[sampled_indices]

        normals = normals * movable_mask.unsqueeze(-1)

        flows = self.model(
            points.unsqueeze(0),
            movable_mask.unsqueeze(0).unsqueeze(-1),
            normals.unsqueeze(0),
        )[0]  # (N, 3)

        direction = self._post_filtering(
            points,
            movable_mask,
            grasping_point,
            flows,
            num_neighbors=self.num_neighbors,
        )  # (3,)

        return direction.cpu().numpy()

    @torch.inference_mode()
    def _post_filtering(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        grasping_point: torch.Tensor,
        flows: torch.Tensor,
        num_neighbors: int = 16,
    ):
        mask = mask > 0

        tf_samples = self._estimate_sub_part_transformation(
            points, flows, mask, num_neighbors=num_neighbors
        )  # (S, 4, 4)
        if tf_samples is None:
            logging.warning("Failed to estimate sub-part transformation")

            return None

        dir_samples = (
            (tf_samples[..., :3, :3] @ grasping_point.reshape(1, 3, 1)).squeeze(-1)
            + tf_samples[..., :3, 3]
            - grasping_point.reshape(1, 3)
        )  # (S, 3)
        dir_samples /= torch.norm(dir_samples, p=2, dim=-1, keepdim=True).clip(
            min=1e-15
        )

        fb = SteinVMFDistribution().fit(dir_samples)
        dir_mode = fb.mode().reshape(3)  # (3,)

        return dir_mode

    @torch.inference_mode()
    def _estimate_sub_part_transformation(
        self,
        points: torch.Tensor,
        flows: torch.Tensor,
        mask: torch.Tensor,
        num_neighbors: int = 16,
    ):
        points_src = points  # (N, 3)
        points_tgt = points + flows  # (N, 3)

        points_src = points_src[mask]  # (N', 3)
        points_tgt = points_tgt[mask]  # (N', 3)

        if points_src.shape[0] < num_neighbors:
            return None

        nn_idx = knn_points(
            points_src[None, ...],  # (1, N', 3)
            points_tgt[None, ...],  # (1, N', 3)
            K=num_neighbors,
            return_nn=False,
            return_sorted=False,
        )[1].squeeze(0)  # (N', num_neighbors)

        point_src_sample = knn_gather(points_src[None, ...], nn_idx[None, ...]).squeeze(
            0
        )  # (N', num_neighbors, 3)
        point_tgt_sample = knn_gather(points_tgt[None, ...], nn_idx[None, ...]).squeeze(
            0
        )  # (N', num_neighbors, 3)

        rot_samples_transposed, trans_samples, _ = corresponding_points_alignment(
            point_src_sample, point_tgt_sample
        )  # (N', 3, 3), (N', 3), (N',)
        rot_samples = rot_samples_transposed.transpose(-1, -2)  # (N', 3, 3)

        tf_samples = (
            torch.eye(4, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(rot_samples.shape[0], 1, 1)
        )  # (S, 3, 4)
        tf_samples[:, :3, :3] = rot_samples
        tf_samples[:, :3, 3] = trans_samples

        return tf_samples
