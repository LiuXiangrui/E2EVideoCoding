import torch
import torch.nn as nn
import torch.nn.functional as F

from .InterPrediction import MotionEstimation, MotionCompensation


class NonlocalAttentionFVC(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.W = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cur.shape

        x = self.theta(ref).view(batch, channels, -1).permute(0, 2, 1)  # shape (B, H * W, C)
        y = self.phi(cur).view(batch, channels, -1)  # shape (B, C, H * W)

        attn = F.softmax(torch.matmul(x, y), dim=1)  # shape (B, H * W, H * W)

        z = torch.matmul(attn, x).permute(0, 2, 1).view(batch, channels, height, width)

        return self.W(z)


class MultiFrameFeatsFusionBlock(nn.Module):
    def __init__(self, motion_est: nn.Module = None, motion_comp: nn.Module = None):
        super().__init__()
        self.non_local = NonlocalAttentionFVC(in_channels=64)
        self.motion_est = motion_est
        self.motion_comp = motion_comp

    def forward(self, frame: torch.Tensor, ref: torch.Tensor = None) -> torch.Tensor:
        if ref is None:
            assert self.motion_est is None and self.motion_comp is None
            refined_cur = self.non_local(frame, ref=frame)
        else:
            aligned_ref = self.motion_comp(ref=ref, motion_fields=self.motion_est(frame, ref=ref))
            refined_cur = self.non_local(frame, ref=aligned_ref)

        return refined_cur


class MultiFrameFeatsFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_self = MultiFrameFeatsFusionBlock()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation(feats_channels=64, offset_channels=64)

        self.branches_ref = nn.ModuleList([
            MultiFrameFeatsFusionBlock(motion_est=self.motion_est, motion_comp=self.motion_comp),
            MultiFrameFeatsFusionBlock(motion_est=self.motion_est, motion_comp=self.motion_comp),
            MultiFrameFeatsFusionBlock(motion_est=self.motion_est, motion_comp=self.motion_comp)
        ])

        self.fusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)

    def forward(self, feats: torch.Tensor, ref_feats_list: list) -> torch.Tensor:
        while len(ref_feats_list) < 3:  # TODO: 在下次会议提到这一点
            ref_feats_list.append(ref_feats_list[-1].clone())
        feats_list = [self.branch_self(feats), ]
        for ref, branch in zip(ref_feats_list, self.branches_ref):
            feats_list.append(branch(feats, ref=ref))
        fused_feats = self.fusion(torch.cat(feats_list, dim=1))
        return fused_feats
