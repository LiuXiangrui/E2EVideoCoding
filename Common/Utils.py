import argparse
import datetime
import json
import logging
import math
from pathlib import Path

import torch
from prettytable import PrettyTable
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS) -> torch.Tensor:
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))


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
    def __init__(self, item_list: list, truncated_decimal: int = 4) -> None:
        self.item_list = item_list
        self.truncated_decimal = truncated_decimal
        self.data = {n: [0, 0] for n in self.item_list}

    def get_item_list(self) -> list:
        return self.item_list

    def add_item(self, name: str) -> None:
        self.item_list.append(name)
        self.data[name] = [0, 0]

    def update(self, n: str, v: float) -> None:
        self.data[n][0] += v
        self.data[n][1] += 1

    def get(self, n: str, average: bool = False) -> float:
        return self.data[n][0] if average else self.data[n][0] / self.data[n][1]

    def clean(self) -> None:
        for n in self.item_list:
            self.data[n] = [0, 0]

    def display(self) -> str:
        disp = PrettyTable(field_names=self.item_list)
        disp.add_row([round(self.data[n][0] / self.data[n][1], ndigits=self.truncated_decimal) for n in self.item_list])
        info = disp.get_string()
        self.clean()
        return info


class Arguments:
    def __init__(self, args: dict) -> None:
        self.parse_args(args=args)

    def parse_args(self, args: dict) -> None:
        for key, value in args.items():
            if isinstance(value, dict):
                self.parse_args(value)
            else:
                self.__dict__[key] = value


def init() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="filepath of configuration files")

    with open(parser.parse_args().config, mode='r') as f:
        args = Arguments(args=json.load(f))

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
        tensorboard = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)
        logger.info(r"===========Save tensorboard and logger to {0}===========".format(str(tb_dir)))
    else:
        print(r"===========Disable logger to accelerate training===========")
        logger = None
        tensorboard = None
    return args, logger, ckpt_dir, tensorboard


class DecodedFrameBuffer:
    def __init__(self) -> None:
        super().__init__()
        self.frame_buffer = list()

    def get_frames(self, num_frames: int = 1) -> list:
        assert 1 <= num_frames <= len(self.frame_buffer)
        return self.frame_buffer[-num_frames:]

    def update(self, frame: torch.Tensor) -> None:
        self.frame_buffer.append(frame)

    def __len__(self) -> int:
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


def cal_psnr(distortion: torch.Tensor) -> torch.Tensor:
    psnr = -10 * torch.log10(distortion)
    return psnr


def separate_aux_and_normal_params(net: nn.Module, exclude_net: nn.Module = None) -> tuple:
    parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)
    fixed_parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and not p.requires_grad)

    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters | fixed_parameters
    params_dict = dict(net.named_parameters())

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if exclude_net is not None:
        excluded_parameters = set("post_processing." + n for n, p in exclude_net.named_parameters())
        parameters = parameters - (parameters & excluded_parameters)

    params = (params_dict[n] for n in sorted(list(parameters)))
    aux_params = (params_dict[n] for n in sorted(list(aux_parameters)))

    return params, aux_params


def optical_flow_warp(x: torch.Tensor, motion_fields: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = motion_fields.shape
    axis_hor = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(batch, -1, height, -1)
    axis_ver = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(batch, -1, -1, width)
    grid = torch.cat([axis_hor, axis_ver], dim=1).to(motion_fields.device)

    motion_fields = torch.cat([motion_fields[:, 0:1, :, :] / ((width - 1.0) / 2.0),
                               motion_fields[:, 1:2, :, :] / ((height - 1.0) / 2.0)], dim=1)

    warped_x = F.grid_sample(input=x, grid=(grid + motion_fields).permute(0, 2, 3, 1),
                             mode="bilinear", padding_mode="border", align_corners=False)
    return warped_x
