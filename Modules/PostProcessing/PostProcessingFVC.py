import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlocalAttentionFVC(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.W = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        B, C, H, W = cur.shape

        x = self.theta(ref).view(B, C, -1).permute(0, 2, 1)  # shape (B, H * W, C)
        y = self.phi(cur).view(B, C, -1)  # shape (B, C, H * W)

        attn = F.softmax(torch.matmul(x, y), dim=1)  # shape (B, H * W, H * W)

        z = torch.matmul(attn, x).permute(0, 2, 1).view(B, C, H, W)

        return self.W(z)


class MultiFrameFeatsFusionBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.non_local = NonlocalAttentionFVC(in_channels=64)

        self.motion_est = kwargs["motion_est"] if "motion_est" in kwargs.keys() else None
        self.motion_comp = kwargs["motion_comp"] if "motion_comp" in kwargs.keys() else None

    def forward(self, cur: torch.Tensor, **kwargs) -> torch.Tensor:
        if "ref" in kwargs.keys():
            print("POST PROCESSING")
            print(kwargs["ref"].shape)
            offset = self.motion_est(cur, ref=kwargs["ref"])
            print(offset.shape)

            aligned_ref = self.motion_comp(offset, ref=kwargs["ref"])
            refined_cur = self.non_local(cur, ref=aligned_ref)
        else:
            refined_cur = self.non_local(cur, ref=cur)
        return refined_cur


class MultiFrameFeatsFusionFVC(nn.Module):
    def __init__(self, motion_est: nn.Module, motion_comp: nn.Module):
        super().__init__()
        self.branch_self = MultiFrameFeatsFusionBlock()
        self.branches_ref = nn.ModuleList([
            MultiFrameFeatsFusionBlock(motion_est=motion_est, motion_comp=motion_comp),
            MultiFrameFeatsFusionBlock(motion_est=motion_est, motion_comp=motion_comp),
            MultiFrameFeatsFusionBlock(motion_est=motion_est, motion_comp=motion_comp)
        ])

        self.fusion = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)

    def forward(self, cur: torch.Tensor, ref_list: list) -> torch.Tensor:
        feats_list = [self.branch_self(cur), ]
        for ref, branch in zip(ref_list, self.branches_ref):
            feats_list.append(branch(cur, ref=ref))
        fused_cur = self.fusion(torch.cat(feats_list, dim=1))
        return fused_cur
