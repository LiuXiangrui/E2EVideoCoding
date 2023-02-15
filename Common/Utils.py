import argparse
import datetime
import logging
import math
from pathlib import Path
from typing import Union

import torch
from prettytable import PrettyTable
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_weight", type=float, help="weights for rd cost")
    parser.add_argument("--gpu", action='store_true', default=False, help="use gpu or cpu")

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    parser.add_argument("--intra_quality", type=int, default=3, help="quality factor of I frame codec")

    parser.add_argument("--dataset_root", type=str, help="training h5 file")

    parser.add_argument("--checkpoints", type=str, help="checkpoints path")
    parser.add_argument("--pretrained", type=str, help="pretrained model path")

    parser.add_argument("--epoch_milestone", type=list, default=[500, ], help="training epochs per stage")
    parser.add_argument("--lr_milestone", type=list, default=[1e-4, ], help="learning rate for per stage")
    parser.add_argument("--lr_decay_milestone", type=list, default=[100, 200, 300], help="learning rate decay milestone")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="learning rate decay factor")

    parser.add_argument("--eval_epochs", type=int, default=1, help="interval of epochs for evaluation")
    parser.add_argument("--save_epochs", type=int, default=5, help="interval of epochs for model saving")
    parser.add_argument("--save_dir", type=str, default="./Experiments", help="directory for recording")
    parser.add_argument("--verbose", action='store_true', default=False, help="use tensorboard and logger")

    args = parser.parse_args()
    return args


class CustomLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(str(log_dir) + '/Log.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger

    def info(self, msg: str, print_: bool = True) -> None:
        self.logger.info(msg)
        if print_:
            print(msg)


class Record:
    def __init__(self, name: list, truncated_decimal: int = 4) -> None:
        self.name = name
        self.truncated_decimal = truncated_decimal
        self.data = {n: [0, 0] for n in self.name}
        self.disp = PrettyTable(field_names=name)

    def update(self, n: str, v: float) -> None:
        self.data[n][0] += v
        self.data[n][1] += 1

    def get(self, n: str, average: bool = False) -> float:
        return self.data[n][0] if average else self.data[n][0] / self.data[n][1]

    def clean(self) -> None:
        for n in self.name:
            self.data[n] = [0, 0]

    def display(self) -> str:
        self.disp.add_row([round(self.data[n][0] / self.data[n][1], ndigits=self.truncated_decimal) for n in self.name])
        info = self.disp.get_string()
        self.disp.clear_rows()
        self.clean()
        return info


def init():
    args = parse_args()
    experiment_dir = Path(args.save_dir)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)
    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)
    print(r"===========Save checkpoints to {0}===========".format(str(ckpt_dir)))
    if args.verbose:
        log_dir = experiment_dir.joinpath('Log/')
        logger = CustomLogger(log_dir=log_dir)
        logger.info('PARAMETER ...', print_=False)
        logger.info(str(args), print_=False)
        tb_dir = experiment_dir.joinpath('Tensorboard/')
        tb_dir.mkdir(exist_ok=True)
        tb = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)
        logger.info(r"===========Save tensorboard and logger to {0}===========".format(str(tb_dir)))
    else:
        print(r"===========Disable logger to accelerate training===========")
        logger = None
        tb = None
    return args, logger, ckpt_dir, tb


class DecodedBuffer:
    def __init__(self, require_feats_buffer: bool = False):
        super().__init__()
        self.frame_buffer = list()
        self.feats_buffer = list() if require_feats_buffer else None

    def get_frames(self, num_frames: int = 1) -> Union[list, torch.Tensor]:
        assert 1 <= num_frames <= len(self.frame_buffer)
        return self.frame_buffer[-num_frames:] if num_frames > 1 else self.frame_buffer[-1]

    def get_feats(self, num_feats: int = 1) -> Union[list, torch.Tensor]:
        assert self.feats_buffer is not None and 1 <= num_feats <= len(self.feats_buffer)
        return self.feats_buffer[-num_feats:] if num_feats > 1 else self.feats_buffer[-1]

    def update(self, frame: torch.Tensor, **kwargs):
        self.frame_buffer.append(frame)
        if self.feats_buffer is not None and "feats" in kwargs.keys():
            self.feats_buffer.append(kwargs["feats"])

    def __len__(self):
        return len(self.frame_buffer)


def calculate_bpp(likelihoods: torch.Tensor | dict, num_pixels: int) -> torch.Tensor:
    if isinstance(likelihoods, torch.Tensor):
        return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    elif isinstance(likelihoods, dict):
        bpp = torch.zeros(1)
        for k, v in likelihoods.items():
            assert isinstance(v, torch.Tensor)
            bpp = bpp.to(v.device)
            bpp += torch.log(v).sum() / (-math.log(2) * num_pixels)
        return bpp


def cal_psnr(distortion: torch.Tensor):
    psnr = -10 * torch.log10(distortion)
    return psnr


def separate_aux_normal_params(net: nn.Module):
    parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)
    fixed_parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and not p.requires_grad)

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters | fixed_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    params = (params_dict[n] for n in sorted(list(parameters)))
    aux_params = (params_dict[n] for n in sorted(list(aux_parameters)))

    return params, aux_params


def optical_flow_warp(x: torch.Tensor, motion_fields: torch.Tensor) -> torch.Tensor:
    B, C, H, W = motion_fields.shape
    axis_hor = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    axis_ver = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([axis_hor, axis_ver], dim=1).to(motion_fields.device)

    motion_fields = torch.cat([motion_fields[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                               motion_fields[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)

    warped_x = F.grid_sample(input=x, grid=(grid + motion_fields).permute(0, 2, 3, 1),
                             mode="bilinear", padding_mode="border", align_corners=False)
    return warped_x
