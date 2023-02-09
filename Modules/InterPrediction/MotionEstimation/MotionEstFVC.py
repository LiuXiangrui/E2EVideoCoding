import torch
import torch.nn as nn


class MotionEstFVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        offset = self.net(torch.cat([cur, ref], dim=1))
        return offset
