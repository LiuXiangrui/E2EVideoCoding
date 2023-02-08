import torch
from torch.nn import functional as F


def optical_flow_warp(x: torch.Tensor, optical_flow: torch.Tensor) -> torch.Tensor:
    B, C, H, W = optical_flow.shape
    axis_hor = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    axis_ver = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([axis_hor, axis_ver], dim=1).to(optical_flow.device)

    optical_flow = torch.cat([optical_flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              optical_flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)

    warped_x = F.grid_sample(input=x, grid=(grid + optical_flow).permute(0, 2, 3, 1),
                             mode="bilinear", padding_mode="border", align_corners=False)
    return warped_x
