import torch
import torch.nn as nn

from Modules import MotionCompensation, MotionEstimation
from Modules import ContextualCompression, MotionCompression


class InterFrameCodecDCVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.contextual_compression = ContextualCompression()
        self.motion_compression = MotionCompression()

    def inter_predict(self, frame: torch.Tensor, ref: torch.Tensor):
        motion_fields = self.motion_est(frame, ref=ref)
        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]
        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        pred = torch.clamp(pred, min=0.0, max=1.0)

        return pred, motion_likelihoods

    def frame_compress(self, frame: torch.Tensor, pred: torch.Tensor):
        enc_results = self.contextual_compression(frame, ctx=pred)
        frame_hat = enc_results["x_hat"]
        frame_likelihoods = enc_results["likelihoods"]
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat, frame_likelihoods

    def forward(self, frame: torch.Tensor, ref: torch.Tensor):
        pred, motion_likelihoods = self.inter_predict(frame, ref=ref)
        frame_hat, frame_likelihoods = self.frame_compress(frame, pred=pred)

        return frame_hat, pred, motion_likelihoods, frame_likelihoods


if __name__ == "__main__":
    a = InterFrameCodecDCVC()
    b = a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
