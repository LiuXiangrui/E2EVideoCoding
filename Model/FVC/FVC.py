import torch
import torch.nn as nn

from Model.Common.BasicBlock import EncUnit, DecUnit
from .Modules import MotionCompensation, MotionEstimation
from .Modules import ResiduesCompression, MotionCompression


class InterFrameCodecFVC(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()
        self.feats_extraction = EncUnit(in_channels=3, out_channels=64)
        self.motion_est = MotionEstimation(feats_channels=64)
        self.motion_comp = MotionCompensation(feats_channels=64, offset_channels=128)
        self.motion_compression = MotionCompression(N=network_config["N_motion"], M=network_config["M_motion"])
        self.residues_compression = ResiduesCompression(N=network_config["N_residues"], M=network_config["M_residues"])
        self.frame_reconstruction = DecUnit(in_channels=64, out_channels=3)

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

        frame_hat = self.frame_reconstruction(feats_hat)

        return frame_hat, residues_likelihoods, motion_likelihoods

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.residues_compression.aux_loss()

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        feats = self.feats_extraction(frame)
        ref_feats = self.feats_extraction(ref)

        motion_fields = self.motion_est(feats, ref=ref_feats)
        motion_enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = motion_enc_results["strings"]
        motion_hyper_shape = motion_enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]
        pred = self.motion_comp(ref_feats, motion_fields=motion_fields_hat)

        residues = feats - pred
        frame_enc_results = self.residues_compression.compress(residues)

        return motion_enc_results, frame_enc_results

    @torch.no_grad()
    def decode(self, ref: torch.Tensor, motion_dec_results: dict, frame_dec_results: dict) -> torch.Tensor:
        ref_feats = self.feats_extraction(ref)

        motion_strings = motion_dec_results["strings"]
        motion_hyper_shape = motion_dec_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]
        pred = self.motion_comp(ref_feats, motion_fields=motion_fields_hat)

        frame_strings = frame_dec_results["strings"]
        frame_hyper_shape = frame_dec_results["shape"]
        residues_hat = self.residues_compression.decompress(strings=frame_strings, shape=frame_hyper_shape)["x_hat"]

        feats_hat = pred + residues_hat
        frame_hat = self.frame_reconstruction(feats_hat)
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        super().load_state_dict(state_dict=state_dict, strict=strict)
        if not self.training:
            self.motion_compression.update()
            self.residues_compression.update()
