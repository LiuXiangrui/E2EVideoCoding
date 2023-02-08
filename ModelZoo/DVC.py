import torch
import torch.nn as nn

from Modules.PreTransform import IdentityTransform
from Modules.InterPrediction.MotionEstimation.MotionEstDVC import MotionEstDVC
from Modules.InterPrediction.MotionCompensation.MotionCompDVC import MotionCompDVC
from Modules.Codec.FrameCodecDVC import FrameCodecDVC
from Modules.Codec.MotionCodecDVC import MotionCodecDVC


class InterFrameCodecDVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_transform = IdentityTransform()
        self.motion_est = MotionEstDVC()
        self.motion_comp = MotionCompDVC()
        self.frame_codec = FrameCodecDVC()
        self.motion_codec = MotionCodecDVC()

    def forward(self, cur_frame: torch.Tensor, ref_frame: torch.Tensor):
        cur_frame = self.pre_transform(cur_frame)
        offset = self.motion_est(cur_frame, ref_frame=ref_frame)
        offset_encode_results = self.motion_codec(offset)
        pred_frame = self.motion_comp(ref_frame, rec_offset=offset_encode_results["rec_offset"])
        frame_encode_results = self.frame_codec(cur_frame, pred_frame=pred_frame)
        return {
            "rec_offset": offset_encode_results["rec_offset"],
            "rec_frame": frame_encode_results["rec_frame"],
            "likelihoods": {"offset": offset_encode_results["likelihoods"],
                            "frame_latents": frame_encode_results["likelihoods"]["latents"],
                            "frame_hyperprior": frame_encode_results["likelihoods"]["hyperprior"],
                            }
        }

if __name__ == "__main__":
    a = InterFrameCodecDVC()
    b = a(torch.randn(1,3, 128, 128), torch.randn(1, 3, 128, 128))
    print(b)