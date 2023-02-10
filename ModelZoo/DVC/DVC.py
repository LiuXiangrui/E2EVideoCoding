import torch
import torch.nn as nn

from Modules.PreTransform import IdentityPreTransform
from Modules.PostTransform import IdentityPostTransform
from ModelZoo.DVC.InterPrediction import MotionCompensation, MotionEstimation
from ModelZoo.DVC.FrameCompression import FrameCompression
from ModelZoo.DVC.MotionCompression import MotionCompression


class InterFrameCodecDVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_transform = IdentityPreTransform()
        self.motion_est = MotionEstimation()
        self.motion_comp = MotionCompensation()
        self.frame_codec = FrameCompression()
        self.motion_codec = MotionCompression()
        self.post_transform = IdentityPostTransform()

    def forward(self, cur_frame: torch.Tensor, ref_frame: torch.Tensor):
        cur_frame = self.pre_transform(cur_frame)
        offset = self.motion_est(cur_frame, ref=ref_frame)
        offset_encode_results = self.motion_codec(offset)
        rec_offset = offset_encode_results["inputs_hat"]
        pred_frame = self.motion_comp(ref_frame, offset=rec_offset)
        frame_encode_results = self.frame_codec(cur_frame, pred=pred_frame)
        rec_frame = self.post_transform(frame_encode_results["inputs_hat"])
        return {
            "rec_offset": rec_offset,
            "rec_frame": rec_frame,
            "likelihoods": {"offset": offset_encode_results["likelihoods"], "frame": frame_encode_results["likelihoods"]}
        }


if __name__ == "__main__":
    a = InterFrameCodecDVC()
    b = a(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
    print(b)
