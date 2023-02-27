import argparse
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ops import compute_padding
from compressai.zoo import cheng2020_anchor as IntraFrameCodec

from Model.Common.Utils import Arguments, DecodedFrameBuffer
from Model.DCVC import InterFrameCodecDCVC
from Model.DVC import InterFrameCodecDVC
from Model.FVC import InterFrameCodecFVC

torch.backends.cudnn.deterministic = True

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
    "num_frames": 16,
    "quality_idx": 8,
    "gop_size": 8,
}

quality_idx_to_network_cfg_map = {
    "DVC": {
        1: {"N_motion": 128, "M_motion": 192, "N_residues": 128, "M_residues": 192},
        2: {"N_motion": 128, "M_motion": 192, "N_residues": 128, "M_residues": 192},
        3: {"N_motion": 128, "M_motion": 192, "N_residues": 128, "M_residues": 192},
        4: {"N_motion": 128, "M_motion": 192, "N_residues": 128, "M_residues": 192}
    },
    "DCVC": {  # TODO: need modified
        1: {"N_motion": 128, "M_motion": 192, "N_frame": 128, "M_frame": 192},
        2: {"N_motion": 128, "M_motion": 192, "N_frame": 128, "M_frame": 192},
        3: {"N_motion": 128, "M_motion": 192, "N_frame": 128, "M_frame": 192},
        4: {"N_motion": 128, "M_motion": 192, "N_frame": 128, "M_frame": 192}
    },
}


class Encoder:
    def __init__(self, gop_size: int, quality_idx: int, gpu: bool, intra_frame_codec: nn.Module, inter_frame_codec: nn.Module, network_cfg_map: dict, ckpt_folder: str):
        super().__init__()

        self.gop_size = gop_size
        self.quality_idx = quality_idx
        self.device = "cuda" if gpu else "cpu"

        self.intra_frame_codec = intra_frame_codec(quality=quality_idx_map[quality_idx][1], metric="mse", pretrained=True)
        self.inter_frame_codec = inter_frame_codec(network_config=network_cfg_map[quality_idx])

        self.intra_frame_codec.to(self.device)
        self.inter_frame_codec.to(self.device)
        self.intra_frame_codec.eval()
        self.inter_frame_codec.eval()

        inter_model_path = os.path.join(ckpt_folder, "{}.pth".format(quality_idx_map[quality_idx][0]))

        self.inter_frame_codec.load_state_dict(torch.load(inter_model_path, map_location=self.device)["inter_frame_codec"])

    @torch.no_grad()
    def compress_sequence(self, seq_path: str, height: int, width: int, num_frames: int, bin_path: str) -> float:
        with open(bin_path, mode='wb') as f:
            # load yuv sequence and convert to rgb frames
            frames = self.load_yuv(seq_path=seq_path, height=height, width=width, num_frames=num_frames)

            # write header bitstream to file
            self.write_header_stream(f, frames)

            # note that number of pixels need to calculate before padding
            num_pixels = frames[0].shape[-2] * frames[0].shape[-1] * len(frames)

            # pad to 64x
            frames = [self.pad_frame_to_64x(frame=frame) for frame in frames]

            enc_results = []
            # compress gop
            gop_frames = [frames[i: i + self.gop_size - 1] for i in range(0, len(frames), self.gop_size)]
            for frames in gop_frames:
                enc_results.extend(self.compress_gop(frames))
            # write bitstream to file
            for results in enc_results:
                self.write_frame_stream(results, f=f)

        bpp = self.calculate_bpp(bin_path=bin_path, num_pixels=num_pixels)

        return bpp

    @torch.no_grad()
    def compress_gop(self, frames: list) -> list:
        decode_frame_buffer = DecodedFrameBuffer()

        enc_results_list = []

        # compress intra frame
        intra_frame = frames[0]
        enc_results = self.intra_frame_codec.compress(intra_frame)
        enc_results_list.append(enc_results)

        # decompress intra frame and add it to decoded buffer
        intra_frame_hat = self.intra_frame_codec.decompress(strings=enc_results["strings"], shape=enc_results["shape"])["x_hat"]
        intra_frame_hat = torch.clamp(intra_frame_hat, min=0., max=1.)
        decode_frame_buffer.update(intra_frame_hat)

        for frame in frames[1:]:
            ref = decode_frame_buffer.get_frames(num_frames=1)[0]
            motion_enc_results, frame_enc_results = self.inter_frame_codec.encode(frame, ref=ref)
            frame_hat = self.inter_frame_codec.decode(ref=ref, motion_enc_results=motion_enc_results, frame_enc_results=frame_enc_results)
            decode_frame_buffer.update(frame_hat)

            enc_results_list.append(motion_enc_results)
            enc_results_list.append(frame_enc_results)

        return enc_results_list

    def write_header_stream(self, f, frames) -> None:
        num_frames = int(len(frames))
        height = int(frames[0].shape[-2])
        width = int(frames[0].shape[-1])

        self.write_uintx(f, value=self.gop_size, x=header_info_bit_depth_map["gop_size"])
        self.write_uintx(f, value=self.quality_idx, x=header_info_bit_depth_map["quality_idx"])
        self.write_uintx(f, value=num_frames, x=header_info_bit_depth_map["num_frames"])
        self.write_uintx(f, value=width, x=header_info_bit_depth_map["width"])
        self.write_uintx(f, value=height, x=header_info_bit_depth_map["height"])

    def write_frame_stream(self, enc_results: dict, f) -> None:
        shape = enc_results["shape"]
        strings = enc_results["strings"]

        for value in shape:  # write the shape of features needed to be decoded by entropy bottleneck
            self.write_uintx(f, value=value, x=16)

        num_string = len(strings)
        self.write_uintx(f, value=num_string, x=8)  # write how many strings need to write

        for string in strings:
            string = string[0]  # note that string is a list containing 1 element, and I don't know why?
            len_string = len(string)
            self.write_uintx(f, value=len_string, x=32)  # write the length of the string
            self.write_bytes(f, values=string)  # write the string

    @staticmethod
    def pad_frame_to_64x(frame: torch.Tensor) -> torch.Tensor:
        _, _, h, w = frame.shape
        pad, _ = compute_padding(h, w, min_div=64)
        return F.pad(frame, pad, mode="constant", value=0)

    def load_yuv(self, seq_path: str, height: int, width: int, num_frames: int) -> list:
        yuv_frames = self.read_yuv_420p(yuv_filepath=seq_path, height=height, width=width, num_frames=num_frames)
        rgb_frames = [self.yuv420_to_rgb(yuv_frame=yuv_frame) for yuv_frame in yuv_frames]
        frames = [torch.from_numpy(frame / 255.).permute(1, 2, 0).unsqueeze(dim=1) for frame in rgb_frames]
        return frames

    @staticmethod
    def read_yuv_420p(yuv_filepath: str, height: int, width: int, num_frames: int) -> list:
        frames = []

        chroma_height = height // 2
        chroma_width = width // 2
        with open(yuv_filepath, mode='rb') as f:
            frame_counter = 0
            while frame_counter < num_frames:
                y_data = np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8)
                u_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                    (chroma_height, chroma_width)).astype(np.uint8)
                v_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                    (chroma_height, chroma_width)).astype(np.uint8)
                frames.append([y_data, u_data, v_data])
                frame_counter += 1
        return frames

    @staticmethod
    def yuv420_to_rgb(yuv_frame: list) -> np.ndarray:
        """
        color space conversion
        :param yuv_frame: input yuv frame [y, u v], each component is stored in a numpy array
        :return rgb_frame: frame in rgb color space
        """
        convert_matrix = np.array([
            [1.000, 1.000, 1.000],
            [0.000, -0.394, 2.032],
            [1.140, -0.581, 0.000],
        ])

        y_data, u_data, v_data = yuv_frame

        # pad chroma data to the size of luma data
        u_data = np.repeat(np.repeat(u_data, repeats=2, axis=0), repeats=2, axis=1)
        v_data = np.repeat(np.repeat(v_data, repeats=2, axis=0), repeats=2, axis=1)

        yuv_frame = np.stack([y_data, u_data, v_data], axis=-1).astype(np.float64)
        yuv_frame[:, :, 1:] -= 127.5

        rgb_frame = np.dot(yuv_frame, convert_matrix)

        rgb_frame = np.clip(rgb_frame, a_min=0., a_max=255.).astype(np.uint8)

        return rgb_frame

    @staticmethod
    def calculate_bpp(bin_path: str, num_pixels: int):
        bits = os.path.getsize(bin_path) * 8
        bpp = bits / num_pixels
        return bpp

    @staticmethod
    def write_uintx(f, value: int, x: int) -> None:
        bit_depth_map = {
            8: 'B',
            16: 'H',
            32: 'I'
        }
        f.write(struct.pack(">{}".format(bit_depth_map[x]), value))

    @staticmethod
    def write_bytes(f, values, fmt=">{:d}s"):
        if len(values) == 0:
            return
        f.write(struct.pack(fmt.format(len(values)), values))
        return len(values) * 1


class Decoder:
    def __init__(self, gpu: bool, intra_frame_codec: nn.Module, inter_frame_codec: nn.Module, network_cfg_map: dict, ckpt_folder: str):
        self.device = "cuda" if gpu else "cpu"
        self.ckpt_folder = ckpt_folder
        self.network_cfg_map = network_cfg_map
        self.intra_frame_codec = intra_frame_codec
        self.inter_frame_codec = inter_frame_codec

    @torch.no_grad()
    def decompress_sequence(self, bin_path: str, rec_path: str):
        with open(bin_path, mode='rb') as f:
            # decompress header bitstream
            header_info = self.read_header_stream(f)
            quality_idx = header_info["quality_idx"]
            gop_size = header_info["gop_size"]
            num_frames = header_info["num_frames"]
            height = header_info["height"]
            width = header_info["width"]

            # initialize network and load model weights
            self.intra_frame_codec = self.intra_frame_codec(quality=quality_idx_map[quality_idx][1], metric="mse", pretrained=True)
            self.inter_frame_codec = self.inter_frame_codec(network_config=self.network_cfg_map[quality_idx])

            inter_model_path = os.path.join(self.ckpt_folder, "{}.pth".format(quality_idx_map[quality_idx][0]))
            self.inter_frame_codec.load_state_dict(torch.load(inter_model_path, map_location=self.device)["inter_frame_codec"])

            self.intra_frame_codec.to(self.device)
            self.inter_frame_codec.to(self.device)
            self.intra_frame_codec.eval()
            self.inter_frame_codec.eval()

            # decompress frame bitstream
            decoded_results = [self.read_frame_stream(f) for _ in range(num_frames)]

        # decompress frames from bitstream
        decoded_frames = []
        gop_results = [decoded_results[i: i + gop_size - 1] for i in range(0, num_frames, gop_size)]
        for results in gop_results:
            decoded_frames.extend(self.decompress_gop(results))

        # crop to original size
        decoded_frames = [self.crop_to_origin_size(frame, ori_height=height, ori_width=width) for frame in decoded_frames]

        self.save_yuv(decoded_frames=decoded_frames, rec_path=rec_path)

    @torch.no_grad()
    def decompress_gop(self, dec_results_list) -> list:
        decoded_frame_buffer = DecodedFrameBuffer()

        # decompress intra frame
        dec_results_intra = dec_results_list[0]

        dec_results = self.intra_frame_codec.decompress(strings=dec_results_intra["strings"], shape=dec_results_intra["shape"])
        intra_frame_hat = torch.clamp(dec_results["x_hat"], min=0., max=1.)
        decoded_frame_buffer.update(intra_frame_hat)

        dec_results_list = dec_results_list[1:]
        for frame_idx in range(dec_results_list // 2):
            ref = decoded_frame_buffer.get_frames(num_frames=1)[0]
            motion_enc_results = dec_results_list[frame_idx]
            frame_enc_results = dec_results_list[
                                                     frame_idx + 1]
            frame_hat = self.inter_frame_codec.decode(ref=ref, motion_enc_results=motion_enc_results, frame_enc_results=frame_enc_results)
            decoded_frame_buffer.update(frame_hat)

        decoded_frames = decoded_frame_buffer.get_frames(num_frames=len(decoded_frame_buffer))

        return decoded_frames

    def read_header_stream(self, f) -> dict:
        gop_size = self.read_uintx(f, x=header_info_bit_depth_map["gop_size"])
        quality_idx = self.read_uintx(f, x=header_info_bit_depth_map["quality_idx"])
        num_frames = self.read_uintx(f, x=header_info_bit_depth_map["num_frames"])
        width = self.read_uintx(f, x=header_info_bit_depth_map["width"])
        height = self.read_uintx(f, x=header_info_bit_depth_map["height"])

        head_info = {
            "gop_size": gop_size,
            "quality_idx": quality_idx,
            "num_frames": num_frames,
            "width": width,
            "height": height
        }

        return head_info

    def read_frame_stream(self, f) -> dict:
        # read the shape of features needed to be decoded by entropy bottleneck
        shape = [self.read_uintx(f, x=16) for _ in range(2)]

        num_string = self.read_uintx(f, x=8)  # write how many strings need to write

        strings = [[self.read_bytes(f, self.read_uintx(f, x=32)), ] for _ in range(num_string)]

        return {"strings": strings, "shape": shape}

    @staticmethod
    def crop_to_origin_size(frame: torch.Tensor, ori_height: int, ori_width: int) -> torch.Tensor:
        _, _, h, w = frame.shape
        _, unpad = compute_padding(h, w, out_h=ori_height, out_w=ori_width)
        return F.pad(frame, unpad, mode="constant", value=0)

    def save_yuv(self, decoded_frames: list, rec_path: str):
        # convert rgb frames to yuv frames
        decoded_frames = [self.rgb_to_yuv420(rgb_data=decoded_frame.cpu().numpy() * 255.) for decoded_frame in decoded_frames]
        self.write_yuv420p(decoded_frames=decoded_frames, rec_path=rec_path)

    @staticmethod
    def write_yuv420p(decoded_frames: list, rec_path: str):
        with open(rec_path, mode='wb') as f:
            for decoded_frame in decoded_frames:
                y_data, u_data, v_data = decoded_frame
                np.asarray(y_data, dtype=np.uint8).tofile(f)
                np.asarray(u_data, dtype=np.uint8).tofile(f)
                np.asarray(v_data, dtype=np.uint8).tofile(f)

    @staticmethod
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

    @staticmethod
    def read_uintx(f, x: int) -> int:
        bit_depth_map = {
            8: 'B',
            16: 'H',
            32: 'I'
        }
        return struct.unpack(">{}".format(bit_depth_map[x]), f.read(struct.calcsize(bit_depth_map[x])))[0]

    @staticmethod
    def read_bytes(f, n, fmt=">{:d}s"):
        return struct.unpack(fmt.format(n), f.read(n * struct.calcsize("s")))[0]


def calculate_yuv_psnr(ori_path: str, rec_path: str, height: int, width: int, num_frames: int) -> float:
    y_psnr = u_psnr = v_psnr = 0.
    ori_frames = []
    rec_frames = []

    chroma_height = height // 2
    chroma_width = width // 2
    with open(ori_path, mode='rb') as f:
        frame_counter = 0
        while frame_counter < num_frames:
            y_data = np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8)
            u_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                (chroma_height, chroma_width)).astype(np.uint8)
            v_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                (chroma_height, chroma_width)).astype(np.uint8)
            ori_frames.append([y_data, u_data, v_data])
            frame_counter += 1

    with open(rec_path, mode='rb') as f:
        frame_counter = 0
        while frame_counter < num_frames:
            y_data = np.reshape(np.frombuffer(f.read(height * width), 'B'), (height, width)).astype(np.uint8)
            u_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                (chroma_height, chroma_width)).astype(np.uint8)
            v_data = np.reshape(np.frombuffer(f.read(chroma_height * chroma_width), 'B'),
                                (chroma_height, chroma_width)).astype(np.uint8)
            rec_frames.append([y_data, u_data, v_data])
            frame_counter += 1

    for ori_frame, rec_frame in zip(ori_frames, rec_frames):
        y_mse = np.mean((ori_frame[0] - rec_frame[0]) ** 2)
        y_psnr += np.log10(255 * 255 / y_mse)

        u_mse = np.mean((ori_frame[1] - rec_frame[1]) ** 2)
        u_psnr += np.log10(255 * 255 / u_mse)

        v_mse = np.mean((ori_frame[2] - rec_frame[2]) ** 2)
        v_psnr += np.log10(255 * 255 / v_mse)

    psnr = (6 * y_psnr + u_psnr + v_psnr) / 8

    return psnr


def test_all_classes():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="filepath of configuration files")

    with open(parser.parse_args().config, mode='r') as f:
        args = json.load(f)
        network_args = Arguments(args=args["Network"])
        testing_args = Arguments(args=args["Testing"])

    quality_idx = testing_args.quality_idx

    ckpt_folder = testing_args.ckpt_folder

    intra_frame_codec = IntraFrameCodec
    inter_frame_codec = available_model[network_args.name]

    encoder = Encoder(gop_size=testing_args.gop_size, quality_idx=quality_idx,
                      intra_frame_codec=intra_frame_codec, inter_frame_codec=inter_frame_codec,
                      ckpt_folder=ckpt_folder, gpu=testing_args.gpu, network_cfg_map=quality_idx_to_network_cfg_map[network_args.name])

    decoder = Decoder(intra_frame_codec=intra_frame_codec, inter_frame_codec=inter_frame_codec,
                      ckpt_folder=ckpt_folder, gpu=testing_args.gpu, network_cfg_map=quality_idx_to_network_cfg_map[network_args.name])

    seq_cfg_folder = testing_args.seq_cfg_folder
    compress_folder = testing_args.compress_folder
    results_folder = testing_args.results_folder

    os.makedirs(results_folder, exist_ok=True)

    for class_cfg in os.listdir(seq_cfg_folder):
        class_name = os.path.splitext(class_cfg)[0]
        folder_per_class = os.path.join(compress_folder, class_name)
        os.makedirs(folder_per_class, exist_ok=True)

        rd_record = {}

        with open(os.path.join(seq_cfg_folder, class_cfg), 'r') as f:
            data = json.load(f)
            seq_folder = data["base_path"]
            seq_cfg_dict = data["sequences"]

            for seq_name in seq_cfg_dict.keys():
                seq_path = os.path.join(seq_folder, seq_name + ".yuv")

                rec_seq_path = os.path.join(folder_per_class, seq_name + ".yuv")
                bin_path = os.path.join(folder_per_class, seq_name + ".bin")

                bpp = encoder.compress_sequence(seq_path=seq_path, bin_path=bin_path)

                decoder.decompress_sequence(rec_seq_path=rec_seq_path, bin_path=bin_path)

                psnr = calculate_yuv_psnr(ori_path=seq_path, rec_path=rec_seq_path,
                                          height=seq_cfg_dict[seq_name]["SourceHeight"], width=seq_cfg_dict[seq_name]["SourceWidth"],
                                          num_frames=seq_cfg_dict[seq_name]["FramesToBeEncoded"])

                rd_record[seq_name] = [bpp, psnr]

        average_bpp = sum([rd_record[seq_name][0] for seq_name in rd_record.keys()]) / len(rd_record)
        average_psnr = sum([rd_record[seq_name][1] for seq_name in rd_record.keys()]) / len(rd_record)
        rd_record[class_name] = [average_bpp, average_psnr]

        with open(os.path.join(results_folder, "{}.json".format(class_name)), mode='w') as f:
            json.dump(rd_record, f)


if __name__ == "__main__":
    test_all_classes()
