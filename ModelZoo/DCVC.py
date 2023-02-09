import torch
import torch.nn as nn

from Modules.PreTransform import IdentityTransform
from Modules.InterPrediction.MotionEstimation.MotionEstDCVC import MotionEstDCVC
from Modules.InterPrediction.MotionCompensation.MotionCompDCVC import MotionCompDCVC
from Modules.Codec.FrameCodecDCVC import FrameCodecDCVC
from Modules.Codec.MotionCodecDCVC import MotionCodecDCVC


class InterFrameCodecDCVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_transform = IdentityTransform()
        self.motion_est = MotionEstDCVC()
        self.motion_comp = MotionCompDCVC()
        self.frame_codec = FrameCodecDCVC()
        self.motion_codec = MotionCodecDCVC()

    def forward(self, cur_frame: torch.Tensor, ref_frame: torch.Tensor):
        cur_frame = self.pre_transform(cur_frame)
        offset = self.motion_est(cur_frame, ref_frame=ref_frame)
        offset_encode_results = self.motion_codec(offset)
        temporal_ctx = self.motion_comp(ref_frame, rec_offset=offset_encode_results["rec_offset"])
        frame_encode_results = self.frame_codec(cur_frame, temporal_ctx=temporal_ctx)
        return {
            "rec_offset": offset_encode_results["rec_offset"],
            "rec_frame": frame_encode_results["rec_frame"],
            "likelihoods": {"offset": offset_encode_results["likelihoods"], "frame": frame_encode_results["likelihoods"]}
        }


if __name__ == "__main__":
    a = InterFrameCodecDCVC()
    b = a(torch.randn(1,3, 128, 128), torch.randn(1, 3, 128, 128))
    print(b)