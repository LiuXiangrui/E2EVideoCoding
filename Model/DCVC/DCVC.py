import torch
import torch.nn as nn

from .Modules import MotionCompensation, MotionEstimation
from .Modules import ContextualCompression, MotionCompression
from Model.Common.Utils import optical_flow_warp


class InterFrameCodecDCVC(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation(N=network_config["N_frame"])
        self.motion_compression = MotionCompression(N=network_config["N_motion"], M=network_config["M_motion"])
        self.contextual_compression = ContextualCompression(N=network_config["N_frame"], M=network_config["M_frame"])

    def forward(self, frame: torch.Tensor, ref: torch.Tensor):
        motion_fields = self.motion_est(frame, ref=ref)

        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]

        warped_frame = optical_flow_warp(ref, motion_fields=motion_fields_hat)

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        enc_results = self.contextual_compression(frame, ctx=pred)
        frame_likelihoods = enc_results["likelihoods"]
        frame_hat = enc_results["x_hat"]

        return frame_hat, warped_frame, motion_likelihoods, frame_likelihoods

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.contextual_compression.aux_loss()

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        motion_fields = self.motion_est(frame, ref=ref)

        motion_enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = motion_enc_results["strings"]
        motion_hyper_shape = motion_enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        frame_enc_results = self.contextual_compression.compress(frame, ctx=pred)

        return motion_enc_results, frame_enc_results

    @torch.no_grad()
    def decode(self, ref: torch.Tensor,  motion_enc_results: dict, frame_enc_results: dict) -> torch.Tensor:
        motion_strings = motion_enc_results["strings"]
        motion_hyper_shape = motion_enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

        frame_strings = frame_enc_results["strings"]
        frame_hyper_shape = frame_enc_results["shape"]
        frame_hat = self.contextual_compression.decompress(strings=frame_strings, shape=frame_hyper_shape, ctx=pred)["x_hat"]
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat
