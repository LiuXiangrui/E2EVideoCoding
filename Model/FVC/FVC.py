import torch
import torch.nn as nn

from Model.Common.BasicBlock import EncUnit, DecUnit
from .Modules import MotionCompensation, MotionEstimation
# from Modules import MultiFrameFeatsFusion
from .Modules import ResiduesCompression, MotionCompression


class InterFrameCodecFVC(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()
        self.feats_extraction = EncUnit(in_channels=3, out_channels=64)
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation(feats_channels=64, offset_channels=128)
        self.motion_compression = MotionCompression(N=network_config["N_motion"], M=network_config["M_motion"])
        self.residues_compression = ResiduesCompression(N=network_config["N_residues"], M=network_config["M_residues"])
        self.frame_reconstruction = DecUnit(in_channels=64, out_channels=3)
        # self.post_processing = MultiFrameFeatsFusion()

    def forward(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        feats = self.feats_extraction(frame)
        ref_feats = self.feats_extraction(ref)

        motion_fields = self.motion_est(feats, ref=ref_feats)
        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]
        pred = self.motion_comp(ref_feats, motion_fields=motion_fields_hat)

        residues = feats - pred
        enc_results = self.residues_compression(residues)
        residues_hat = enc_results["x_hat"]
        residues_likelihoods = enc_results["likelihoods"]
        feats_hat = pred + residues_hat

        # if fusion:
        #     feats_hat = self.post_processing(feats, ref_feats_list=ref_feats_list)

        frame_hat = self.frame_reconstruction(feats_hat)

        return frame_hat, residues_likelihoods, motion_likelihoods

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref_frames_list: torch.Tensor) -> tuple:
        feats = self.feats_extraction(frame)
        ref_feats_list = [self.feats_extraction(f) for f in ref_frames_list]

        motion_fields = self.motion_est(feats, ref=ref_feats_list[0])

        enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = enc_results["strings"]
        motion_hyper_shape = enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref_feats_list[0], motion_fields=motion_fields_hat)

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

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.residues_compression.aux_loss()
