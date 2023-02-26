# -- coding: utf-8 --**

import argparse
import datetime
import json
import logging
import math
from pathlib import Path
import struct


import torch
from prettytable import PrettyTable
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
            self.__dict__[key] = Arguments(value) if isinstance(value, dict) else value

    def __str__(self, indent: int = 0) -> str:
        args = ""
        for key, value in self.__dict__.items():
            args += "".join([' '] * indent)
            if isinstance(value, Arguments):
                args += "{}: \n".format(key)
                args += value.__str__(indent=indent + 2)
            else:
                args += "{}: {}\n".format(key, value)
        return args

    def serialize(self):
        return self.__dict__


def init() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="filepath of configuration files")

    with open(parser.parse_args().config, mode='r') as f:
        args = json.load(f)
        network_args = Arguments(args=args["Network"])
        training_args = Arguments(args=args["Training"])

    experiment_dir = Path(training_args.save_directory)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)
    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)
    print(r"===========Save checkpoints to {0}===========".format(str(ckpt_dir)))
    if training_args.verbose:
        log_dir = experiment_dir.joinpath('Log/')
        logger = CustomLogger(log_dir=log_dir)
        logger.info('Network PARAMETER ...\n', print_=False)
        logger.info(str(network_args), print_=False)
        logger.info('Training PARAMETER ...\n', print_=False)
        logger.info(str(training_args), print_=False)
        tb_dir = experiment_dir.joinpath('Tensorboard/')
        tb_dir.mkdir(exist_ok=True)
        tensorboard = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)
        logger.info(r"===========Save tensorboard and logger to {0}===========".format(str(tb_dir)))
    else:
        print(r"===========Disable logger to accelerate training===========")
        logger = None
        tensorboard = None
    return network_args, training_args, logger, ckpt_dir, tensorboard


class DecodedFrameBuffer:
    def __init__(self) -> None:
        super().__init__()
        self.frame_buffer = list()

    def get_frames(self, num_frames: int = 1) -> list:
        assert 1 <= num_frames <= len(self.frame_buffer)
        return self.frame_buffer[-num_frames:]

    def update(self, frame: torch.Tensor) -> None:
        self.frame_buffer.append(frame.clone().detach())

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


def separate_aux_and_normal_params(net: nn.Module, exclude_module_list: list = None) -> tuple:
    parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)
    fixed_parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and not p.requires_grad)

    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters | fixed_parameters
    params_dict = dict(net.named_parameters())

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if exclude_module_list is not None:
        for exclude_module_name in exclude_module_list:
            for n in sorted(list(parameters)):
                if exclude_module_name in n:
                    parameters.remove(n)
            for n in sorted(list(aux_parameters)):
                if exclude_module_name in n:
                    aux_parameters.remove(n)

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


def read_yuv_420p(yuv_filepath: str, height: int, width: int, num_frames: int) -> list:
    frames = []

    chroma_height = height // 2
    chroma_width = width // 2
    with open(yuv_filepath, mode='rb') as f:
        frame_counter = 0
        while frame_counter < num_frames:
            y_data = np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8)
            u_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'), (chroma_height, chroma_width)).astype(np.uint8)
            v_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'), (chroma_height, chroma_width)).astype(np.uint8)
            frames.append([y_data, u_data, v_data])
            frame_counter += 1
    return frames


def yuv420_to_rgb(yuv_data: list) -> np.ndarray:
    convert_matrix = np.array([
        [1.000,  1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])

    y_data, u_data, v_data = yuv_data

    # pad chroma data to the size of luma data
    u_data = np.repeat(np.repeat(u_data, repeats=2, axis=0), repeats=2, axis=1)
    v_data = np.repeat(np.repeat(v_data, repeats=2, axis=0), repeats=2, axis=1)

    yuv_data = np.stack([y_data, u_data, v_data], axis=-1).astype(np.float64)
    yuv_data[:, :, 1:] -= 127.5

    rgb_data = np.dot(yuv_data, convert_matrix)

    rgb_data = np.clip(rgb_data, a_min=0., a_max=255.).astype(np.uint8)

    return rgb_data


def rgb_to_yuv420(rgb_data: np.ndarray) -> list:
    """
    convert rgb array to list of yuv420 array
    :param rgb_data: rgb array with shape (H, W, 3) and the range of data is [0, 255]
    :return yuv_data: list of yuv 420 array and the range of data is [0, 255]
    """
    convert_matrix = np.array([
        [0.29900, -0.147108, 0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400, 0.435912, -0.099978]
    ])

    yuv_data = np.dot(rgb_data, convert_matrix)
    yuv_data[:, :, 1:] += 127.5

    yuv_data = np.clip(yuv_data, a_min=0., a_max=255.).astype(np.uint8)

    y_data, u_data, v_data = np.array_split(yuv_data, indices_or_sections=3, axis=-1)

    yuv_data = [y_data[:, :, 0], u_data[::2, ::2, 0], v_data[::2, ::2, 0]]

    return yuv_data


bit_depth_map = {
        8: 'B',
        16: 'H',
        32: 'I'
}


def write_uintx(f, value: int, bit_depth: int) -> None:
    f.write(struct.pack(">{}".format(bit_depth_map[bit_depth]), value))


def read_uintx(f, bit_depth: int) -> int:
    return struct.unpack(">{}".format(bit_depth_map[bit_depth]), f.read(struct.calcsize(bit_depth_map[bit_depth])))[0]


def write_bytes(f, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    f.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(f, n, fmt=">{:d}s"):
    return struct.unpack(fmt.format(n), f.read(n * struct.calcsize("s")))[0]
