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

    def forward(self, frame: torch.Tensor, ref: torch.Tensor, ref_feats_list: list = None) -> tuple:
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

        if ref_feats_list is not None:
            assert len(ref_feats_list) == 3
            feats_hat = self.post_processing(feats, ref_feats_list=ref_feats_list)

        frame_hat = self.frame_reconstruction(feats_hat)
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat, feats_hat, residues_likelihoods, motion_likelihoods

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        feats = self.feats_extraction(frame)
        ref = self.feats_extraction(ref)

        motion_fields = self.motion_est(feats, ref=ref)

        enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = enc_results["strings"]
        motion_hyper_shape = enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        enc_results = self.residues_compression.compress(feats - pred)
        frame_strings = enc_results["strings"]
        frame_hyper_shape = enc_results["shape"]

        return motion_strings, motion_hyper_shape, frame_strings, frame_hyper_shape

    @torch.no_grad()
    def decode(self, ref: torch.Tensor, motion_strings: list, motion_hyper_shape: list, frame_strings: list, frame_hyper_shape: list, ref_feats_list: list) -> torch.Tensor:
        assert len(ref_feats_list) == 3

        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        residues_hat = self.residues_compression.decompress(strings=frame_strings, shape=frame_hyper_shape)["x_hat"]

        feats_hat = self.post_processing(pred + residues_hat, ref_feats_list=ref_feats_list)
        frame_hat = self.frame_reconstruction(feats_hat)
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat


if __name__ == "__main__":
    a = InterFrameCodecFVC()
    a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128), True, [torch.randn(1, 64, 64, 64)] * 3)
