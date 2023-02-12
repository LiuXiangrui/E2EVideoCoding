import torch
import torch.nn as nn

from Common.BasicBlock import EncUnit, DecUnit

from Modules import MultiFrameFeatsFusion
from Modules import ResiduesCompression, MotionCompression
from Modules import MotionCompensation, MotionEstimation


class InterFrameCodecFVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats_extraction = EncUnit(in_channels=3, out_channels=64)
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.residues_compression = ResiduesCompression()
        self.motion_compression = MotionCompression()
        self.frame_reconstruction = DecUnit(in_channels=64, out_channels=3)
        self.post_processing = MultiFrameFeatsFusion(motion_est=self.motion_est, motion_comp=self.motion_comp)

    def forward(self, frame: torch.Tensor, ref: torch.Tensor, post_processing: bool = False, ref_feats_list: list = None) -> tuple:
        feats = self.feats_extraction(frame)
        ref = self.feats_extraction(ref)

        motion_fields = self.motion_est(feats, ref=ref)
        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]
        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        residues = feats - pred
        enc_results = self.residues_compression(residues)
        residues_hat = enc_results["x_hat"]
        residues_likelihoods = enc_results["likelihoods"]
        feats_hat = pred + residues_hat

        if post_processing:
            feats_hat = self.post_processing(feats, ref_feats_list=ref_feats_list)

        frame_hat = self.frame_reconstruction(feats_hat)

        return frame_hat, residues_likelihoods, motion_likelihoods


if __name__ == "__main__":
    a = InterFrameCodecFVC()
    a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128), True, [torch.randn(1, 64, 64, 64)] * 3)
