from abc import ABCMeta

import torch
from torch import nn as nn

from Modules.Utils import Identity


class MotionCompensationABC(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.offset_process = Identity()
        self.ref_process = Identity()
        self.warp_net = None
        self.refine_net = Identity()

    def forward(self, ref: torch.Tensor, offset: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        processed_ref = self.ref_process(ref)
        kwargs["ref"] = ref
        processed_offset = self.offset_process(offset, **kwargs)
        aligned_ref = self.warp_net(processed_ref, offset=processed_offset)
        kwargs["offset"] = offset  # TODO: need improvement!! so ugly!!!!
        pred = self.refine_net(aligned_ref, *args, **kwargs)
        return pred
