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
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, help="name of method to be tested")
# parser.add_argument("--model_weight", type=str, help="weights of method to be tested")
# parser.add_argument("--gpu", type=bool, action='store_true', default=False, help=" use GPU to test (default = False)")
parser.add_argument("--config", type=str, help="filepath of configuration files")

with open(parser.parse_args().config, mode='r') as f:
    network_args = Arguments(args=json.load(f)["Network"])
    testing_args = Arguments(args=json.load(f)["Testing"])

intra_frame_codec = IntraFrameCodec(quality=testing_args.intra_quality, metric="mse", pretrained=True)
inter_frame_codec = available_model[network_args.network]

record = Record(item_list=["psnr, bpp"])

gop = testing_args.gop

import torch

intra_frame_codec.to("cuda" if testing_args.gpu else "cpu")
inter_frame_codec.to("cuda" if testing_args.gpu else "cpu")
intra_frame_codec.eval()
inter_frame_codec.eval()
@torch.no_grad()
def test_sequences(frames: list):
    pass

@torch.no_grad()
def encode_sequence(frames: list):
    decode_frame_buffer = DecodedFrameBuffer()
    # I frame coding
    intra_frames = frames[::gop]

    for intra_frame in intra_frames:
        intra_frame = intra_frame.to("cuda" if testing_args.gpu else "cpu")

        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]

        enc_results = IntraFrameCodec.compress()


        enc_results = intra_frame_codec(intra_frame)
    intra_frame_hat = torch.clamp(enc_results["x_hat"], min=0.0, max=1.0)
    decode_frame_buffer.update(intra_frame_hat)

    intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
    intra_psnr = cal_psnr(intra_dist)
    intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

    inter_frames = [frames[:, i, :, :, :].to("cuda" if self.training_args.gpu else "cpu") for i in
                    range(1, num_available_frames)]

    alignment = [torch.zeros_like(intra_frame_hat), ]

    rd_cost_avg = aux_loss_avg = align_psnr_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

    # P frame coding
    for frame in inter_frames:
        ref = decode_frame_buffer.get_frames(num_frames=1)[0]

        frame_hat, aligned_ref, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

        align_dist = self.distortion_metric(aligned_ref, frame)
        align_psnr = cal_psnr(align_dist)

        recon_dist = self.distortion_metric(frame_hat, frame)
        recon_psnr = cal_psnr(recon_dist)

        motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
        frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)

        rate = motion_bpp * int(stage == TrainingStage.ME or stage == TrainingStage.ALL) + frame_bpp * int(
            stage == TrainingStage.CONTEXTUAL_CODING or stage == TrainingStage.ALL)
        distortion = align_dist if stage == TrainingStage.ME else recon_dist

        rd_cost = self.training_args.lambda_weight * distortion + rate

        aux_loss = self.inter_frame_codec.aux_loss()

        rd_cost_avg += rd_cost / len(inter_frames)
        aux_loss_avg += aux_loss / len(inter_frames)

        recon_psnr_avg_inter += recon_psnr / len(inter_frames)
        align_psnr_avg += align_psnr / len(inter_frames)

        frame_bpp_avg += frame_bpp / len(inter_frames)
        motion_bpp_avg += motion_bpp / len(inter_frames)

        decode_frame_buffer.update(frame_hat)

        alignment.append(aligned_ref)

    recon_psnr_avg = (recon_psnr_avg_inter * len(inter_frames) + intra_psnr) / num_available_frames

    total_bpp_avg_inter = motion_bpp_avg + frame_bpp_avg
    total_bpp_avg = (total_bpp_avg_inter * len(inter_frames) + intra_bpp) / num_available_frames

    for
