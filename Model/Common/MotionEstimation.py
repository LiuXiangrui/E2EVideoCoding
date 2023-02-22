import torch
import torch.nn as nn
from torch.nn import functional as F

from Model.Common.Utils import optical_flow_warp


class SpyNetBasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpyNetOpticalFlowEst(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_levels = 4
        self.net = nn.ModuleList([SpyNetBasicBlock() for _ in range(self.scale_levels)])
        # self.load_pretrained_model()

    @torch.no_grad()
    def forward(self, cur_frame: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        batch, _, _, _ = cur_frame.shape
        multiscale_cur_frame = [cur_frame.clone(), ]
        multiscale_ref_frame = [ref.clone(), ]
        for level in range(1, self.scale_levels):
            multiscale_cur_frame.append(F.avg_pool2d(multiscale_cur_frame[level - 1], kernel_size=2, stride=2))
            multiscale_ref_frame.append(F.avg_pool2d(multiscale_ref_frame[level - 1], kernel_size=2, stride=2))

        shape_last_scale = multiscale_ref_frame[-1].size()
        optical_flow = torch.zeros(size=[batch, 2, shape_last_scale[2] // 2, shape_last_scale[3] // 2],
                                   dtype=cur_frame.dtype, device=cur_frame.device)
        for level in range(self.scale_levels, 0, -1):
            upsampled_optical_flow = F.interpolate(optical_flow, scale_factor=(2, 2), mode="bilinear") * 2.0
            optical_flow = upsampled_optical_flow + self.net[self.scale_levels - level](torch.cat([
                multiscale_cur_frame[level - 1],
                optical_flow_warp(multiscale_ref_frame[level - 1], upsampled_optical_flow),
                upsampled_optical_flow
            ], dim=1))

        offset = optical_flow
        return offset

    def load_pretrained_model(self):
        model_weights = torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch',
                                                           file_name='spynet-sintel-final')

        model_weights = {key.replace('moduleBasic', 'net'): weight for key, weight in model_weights.items()}
        self.load_state_dict(model_weights, strict=False)
