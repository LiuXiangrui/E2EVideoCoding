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
        with torch.no_grad():
            motion_fields = self.motion_est(frame, ref=ref)
        enc_results = self.motion_compression(motion_fields)
        motion_fields_hat = enc_results["x_hat"]
        motion_likelihoods = enc_results["likelihoods"]
        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)

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

    def aux_loss(self) -> torch.Tensor:
        return self.motion_compression.aux_loss() + self.contextual_compression.aux_loss()

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, ref: torch.Tensor) -> tuple:
        motion_fields = self.motion_est(frame, ref=ref)

        enc_results = self.motion_compression.compress(motion_fields)
        motion_strings = enc_results["strings"]
        motion_hyper_shape = enc_results["shape"]
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]

        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        enc_results = self.contextual_compression.compress(frame, ctx=pred)
        frame_strings = enc_results["strings"]
        frame_hyper_shape = enc_results["shape"]

        return motion_strings, motion_hyper_shape, frame_strings, frame_hyper_shape

    @torch.no_grad()
    def decode(self, ref: torch.Tensor, motion_strings: list, motion_hyper_shape: list, frame_strings: list, frame_hyper_shape: list) -> torch.Tensor:
        motion_fields_hat = self.motion_compression.decompress(strings=motion_strings, shape=motion_hyper_shape)["x_hat"]
        pred = self.motion_comp(ref, motion_fields=motion_fields_hat)
        frame_hat = self.contextual_compression.decompress(strings=frame_strings, shape=frame_hyper_shape, ctx=pred)["x_hat"]
        frame_hat = torch.clamp(frame_hat, min=0.0, max=1.0)

        return frame_hat


if __name__ == "__main__":
    from thop import profile
    a = InterFrameCodecDCVC()
    c, d = profile(a, inputs=(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)))
    print(c)
    print(d)
