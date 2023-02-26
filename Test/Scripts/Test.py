from Model.DVC import InterFrameCodecDVC
from Model.DCVC import InterFrameCodecDCVC
from Model.FVC import InterFrameCodecFVC
from Model.Common.Utils import Arguments, Record, DecodedFrameBuffer
from compressai.zoo import cheng2020_anchor as IntraFrameCodec
import json
import argparse
import torch
available_model = {
    "DVC": InterFrameCodecDVC,
    "DCVC": InterFrameCodecDCVC,
    "FVC": InterFrameCodecFVC
}

quality_idx_map = {
    1: [256, 3],
    2: [512, 4],
    3: [1024, 5],
    4: [2048, 6]
}


header_info_bit_depth_map = {
    "width": 16,
    "height": 16,
    "num_frames": 16,
    "quality_idx": 8,
    "gop_size": 8,
}

from Model.Common.Utils import write_uintx, read_uintx, write_bytes, read_bytes

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, help="name of method to be tested")
# parser.add_argument("--model_weight", type=str, help="weights of method to be tested")
# parser.add_argument("--gpu", type=bool, action='store_true', default=False, help=" use GPU to test (default = False)")
parser.add_argument("--config", type=str, help="filepath of configuration files")

with open(parser.parse_args().config, mode='r') as f:
    network_args = Arguments(args=json.load(f)["Network"])
    testing_args = Arguments(args=json.load(f)["Testing"])


gop = testing_args.gop


quality_idx = testing_args.quality_idx
inter_lambda, intra_quality = quality_idx_map[quality_idx]

intra_frame_codec = IntraFrameCodec(quality=intra_quality, metric="mse", pretrained=True)
inter_frame_codec = available_model[network_args.network]

record = Record(item_list=["psnr, bpp"])

import torch

intra_frame_codec.to("cuda" if testing_args.gpu else "cpu")
inter_frame_codec.to("cuda" if testing_args.gpu else "cpu")
intra_frame_codec.eval()
inter_frame_codec.eval()
@torch.no_grad()
def test_sequences(frames: list):
    pass


def write_header_stream(f, frames) -> None:
    width = int(frames[0].shape[-1])
    height = int(frames[0].shape[-2])
    num_frames = int(len(frames))

    write_uintx(f, value=gop, x=header_info_bit_depth_map["gop_size"])
    write_uintx(f, value=quality_idx, x=header_info_bit_depth_map["quality_idx"])
    write_uintx(f, value=num_frames, x=header_info_bit_depth_map["num_frames"])
    write_uintx(f, value=width, x=header_info_bit_depth_map["width"])
    write_uintx(f, value=height, x=header_info_bit_depth_map["height"])


def read_header_stream(f) -> dict:
    gop_size = read_uintx(f, x=header_info_bit_depth_map["gop_size"])
    quality_idx = read_uintx(f, x=header_info_bit_depth_map["quality_idx"])
    num_frames = read_uintx(f, x=header_info_bit_depth_map["num_frames"])
    width = read_uintx(f, x=header_info_bit_depth_map["width"])
    height = read_uintx(f, x=header_info_bit_depth_map["height"])

    head_info = {
        "gop_size": gop_size,
        "quality_idx": quality_idx,
        "num_frames": num_frames,
        "width": width,
        "height": height
    }

    return head_info


def write_frame_stream(enc_results: dict, f) -> None:
    shape = enc_results["shape"]
    strings = enc_results["strings"]

    for value in shape:  # write the shape of features needed to be decoded by entropy bottleneck
        write_uintx(f, value=value, x=16)

    num_string = len(strings)
    write_uintx(f, value=num_string, x=8)  # write how many strings need to write

    for string in strings:
        string = string[0]  # note that string is a list containing 1 element, and I don't know why?
        len_string = len(string)
        write_uintx(f, value=len_string, x=32)  # write the length of the string
        write_bytes(f, values=string)  # write the string


def read_frame_stream(f) -> dict:
    # read the shape of features needed to be decoded by entropy bottleneck
    shape = [read_uintx(f, x=16) for _ in range(2)]

    num_string = read_uintx(f, x=8)  # write how many strings need to write

    strings = [[read_bytes(f, read_uintx(f, x=32)), ] for _ in range(num_string)]

    return {"strings": strings, "shape": shape}


@torch.no_grad()
def compress_gop(frames: list) -> list:
    decode_frame_buffer = DecodedFrameBuffer()

    enc_results_list = []

    # compress intra frame
    intra_frame = frames[0].to("cuda" if testing_args.gpu else "cpu")
    enc_results = intra_frame_codec.compress(intra_frame)
    enc_results_list.append(enc_results)

    # decompress intra frame and add it to decoded buffer
    dec_results = intra_frame_codec.decompress(strings=enc_results["strings"], shape=enc_results["shape"])
    intra_frame_hat = torch.clamp(dec_results["x_hat"], min=0., max=1.)
    decode_frame_buffer.update(intra_frame_hat)

    for frame in frames[1:]:
        ref = decode_frame_buffer.get_frames()
        motion_enc_results, frame_enc_results = inter_frame_codec.encode(frame, ref=ref)
        frame_hat = inter_frame_codec.decode(ref=ref, motion_enc_results=motion_enc_results, frame_enc_results=frame_enc_results)
        decode_frame_buffer.update(frame_hat)

        enc_results_list.append(motion_enc_results)
        enc_results_list.append(frame_enc_results)

    return enc_results_list


@torch.no_grad()
def compress_sequence(frames: list, bin_path: str) -> None:
    with open(bin_path, mode='wb') as f:
        # write header bitstream to file
        write_header_stream(f, frames)

        enc_results = []
        # compress frames
        gop_frames = [frames[i: i + gop - 1] for i in range(0, len(frames), gop)]
        for frames in gop_frames:
            enc_results.extend(compress_gop(frames))
        # write bitstream to file
        for results in enc_results:
            write_frame_stream(results, f=f)


@torch.no_grad()
def decompress_gop(dec_results_list) -> list:
    decode_frame_buffer = DecodedFrameBuffer()

    # decompress intra frame



@torch.no_grad()
def decompress_sequence(bin_path: str) -> list:
    with open(bin_path, mode='rb') as f:
        # read header bit
        head_info = read_header_stream(f)

        gop_size = head_info["gop_size"]
        num_frames = head_info["num_frames"]
        width = head_info["width"]
        height = head_info["height"]

        frames = [None, ] * num_frames

