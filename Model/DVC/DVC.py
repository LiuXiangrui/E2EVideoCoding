import torch
import torch.nn as nn

from .Modules import MotionCompensation, MotionEstimation
from .Modules import MotionCompression, ResiduesCompression


class InterFrameCodecDVC(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.motion_compression = MotionCompression(N=network_config["N_motion"], M=network_config["M_motion"])
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
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat, aligned_ref, pred, motion_likelihoods, residues_likelihoods

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.residues_compression.aux_loss()

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        motion_fields = self.motion_est(frame, ref=ref)

        enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = enc_results["strings"]
        motion_hyper_shape = enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        _, pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        pred = torch.clamp(pred, min=0.0, max=1.0)

        enc_results = self.residues_compression.compress(frame - pred)
        frame_strings = enc_results["strings"]
        frame_hyper_shape = enc_results["shape"]

        return motion_strings, motion_hyper_shape, frame_strings, frame_hyper_shape

    @torch.no_grad()
    def decode(self, ref: torch.Tensor, motion_strings: list, motion_hyper_shape: list, frame_strings: list, frame_hyper_shape: list) -> torch.Tensor:
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        _, pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        pred = torch.clamp(pred, min=0.0, max=1.0)

        residues_hat = self.residues_compression.decompress(strings=frame_strings, shape=frame_hyper_shape)["x_hat"]

        frame_hat = pred + residues_hat
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat


if __name__ == "__main__":
    a = InterFrameCodecDVC()
    c = a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    exit()
