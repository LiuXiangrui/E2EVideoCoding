import torch
import torch.nn as nn

from Modules import MotionCompensation, MotionEstimation
from Modules import MotionCompression, ResiduesCompression


class InterFrameCodecDVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.residues_compression = ResiduesCompression()
        self.motion_compression = MotionCompression()

    def forward(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        pred, motion_likelihoods = self.inter_predict(frame, ref=ref)
        frame_hat, residues_likelihoods = self.frame_compress(frame, pred=pred)
        return frame_hat, pred, motion_likelihoods, residues_likelihoods

    def inter_predict(self, frame: torch.Tensor, ref: torch.Tensor):
        motion_fields = self.motion_est(frame, ref=ref)
        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]
        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        return pred, motion_likelihoods

    def frame_compress(self, frame: torch.Tensor, pred: torch.Tensor):
        residues = frame - pred
        enc_results = self.residues_compression(residues)
        residues_hat = enc_results["x_hat"]
        residues_likelihoods = enc_results["likelihoods"]
        frame_hat = pred + residues_hat
        return frame_hat, residues_likelihoods


if __name__ == "__main__":
    a = InterFrameCodecDVC()
    c = a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    exit()
