import torch
import torch.nn as nn

from .Modules import MotionCompensation, MotionEstimation
from .Modules import MotionCompression, ResiduesCompression
from Model.Common.MotionEstimation import SpyNetOpticalFlowEst as MotionEstimation
from .Modules.MotionCompensation import OffsetDiversity

class InterFrameCodecDMC(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.motion_compression = MotionCompression(N=network_config["MV_channels"], M=network_config["M_motion"])
        self.residues_compression = ResiduesCompression(N=network_config["N_residues"], M=network_config["M_residues"])

    def forward(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        motion_fields = self.motion_est(frame, ref=ref)

        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]

        aligned_ref, pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        residues = frame - pred

        enc_results = self.residues_compression(residues)
        residues_hat = enc_results["x_hat"]
        residues_likelihoods = enc_results["likelihoods"]

        frame_hat = pred + residues_hat
        frame_hat = torch.clamp(frame_hat, min=0., max=1.)

        return frame_hat, aligned_ref, pred, motion_likelihoods, residues_likelihoods

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.residues_compression.aux_loss()

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        motion_fields = self.motion_est(frame, ref=ref)

        motion_enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = motion_enc_results["strings"]
        motion_hyper_shape = motion_enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        _, pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        pred = torch.clamp(pred, min=0.0, max=1.0)

        frame_enc_results = self.residues_compression.compress(frame - pred)

        return motion_enc_results, frame_enc_results

    @torch.no_grad()
    def decode(self, ref: torch.Tensor, motion_dec_results: dict, frame_dec_results: dict) -> torch.Tensor:
        motion_strings = motion_dec_results["strings"]
        motion_hyper_shape = motion_dec_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        _, pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        pred = torch.clamp(pred, min=0.0, max=1.0)

        frame_strings = frame_dec_results["strings"]
        frame_hyper_shape = frame_dec_results["shape"]
        residues_hat = self.residues_compression.decompress(strings=frame_strings, shape=frame_hyper_shape)["x_hat"]

        frame_hat = pred + residues_hat
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        super().load_state_dict(state_dict=state_dict, strict=strict)
        if not self.training:
            self.motion_compression.update()
            self.residues_compression.update()
