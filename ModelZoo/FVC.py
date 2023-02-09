import torch
import torch.nn as nn

from Modules.PreTransform.PreTransform import FeaturePreTransformFVC
from Modules.PostTransform.PostTransform import FeaturePostTransformFVC
from Modules.InterPrediction.MotionEstimation.MotionEstFVC import MotionEstFVC
from Modules.InterPrediction.MotionCompensation.MotionCompFVC import MotionCompFVC
from Modules.Codec.FrameCodecFVC import FrameCodecFVC
from Modules.Codec.MotionCodecFVC import MotionCodecFVC
from Modules.PostProcessing.PostProcessingFVC import MultiFrameFeatsFusionFVC


class DecodedBuffer:
    def __init__(self):
        super().__init__()
        self.frame_buffer = list()
        self.feats_buffer = list()

    def get_frames(self, num_frames: int = 1) -> list:
        assert 1 <= num_frames <= len(self.frame_buffer)
        return self.frame_buffer[-num_frames:] if num_frames > 1 else self.frame_buffer[-1]

    def get_feats(self, num_feats: int = 1) -> list:
        assert 1 <= num_feats <= len(self.feats_buffer)
        return self.feats_buffer[-num_feats:] if num_feats > 1 else self.feats_buffer[-1]

    def update(self, frame: torch.Tensor, **kwargs):
        if "feats" in kwargs.keys():
            self.frame_buffer.append(frame)
            self.feats_buffer.append(kwargs["feats"])
        else:
            self.frame_buffer.append(frame)
            self.feats_buffer.append(torch.tensor([]))

    def __len__(self):
        return len(self.frame_buffer)


class InterFrameCodecFVC(nn.Module):
    def __init__(self, decoded_buffer: DecodedBuffer):
        super().__init__()
        self.pre_transform = FeaturePreTransformFVC()
        self.motion_est = MotionEstFVC()
        self.motion_comp = MotionCompFVC()
        self.frame_codec = FrameCodecFVC()
        self.motion_codec = MotionCodecFVC()
        self.post_transform = FeaturePostTransformFVC()
        self.post_processing = MultiFrameFeatsFusionFVC(motion_est=self.motion_est, motion_comp=self.motion_comp)
        self.decoded_buffer = decoded_buffer

    def forward(self, cur_frame: torch.Tensor):
        cur_feats = self.pre_transform(cur_frame)
        ref_frame = self.decoded_buffer.get_frames(num_frames=1)
        ref_feats = self.pre_transform(ref_frame)

        offset = self.motion_est(cur_feats, ref=ref_feats)
        offset_encode_results = self.motion_codec(offset)
        pred = self.motion_comp(offset_encode_results["rec_offset"], ref=ref_feats)
        frame_encode_results = self.frame_codec(cur_feats, pred=pred)

        if len(self.decoded_buffer) >= 3:
            rec_feats = self.post_processing(frame_encode_results["rec_cur"], ref_list=self.decoded_buffer.get_feats(num_feats=3))
        else:
            rec_feats = frame_encode_results["rec_cur"]
        rec_frame = self.post_transform(rec_feats)

        self.decoded_buffer.update(rec_frame, feats=rec_feats)

        return {
            "rec_offset": offset_encode_results["rec_offset"],
            "rec_frame": rec_frame,
            "likelihoods": {"offset": offset_encode_results["likelihoods"], "frame": frame_encode_results["likelihoods"]}
        }
