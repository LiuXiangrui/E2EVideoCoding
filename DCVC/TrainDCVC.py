import torch
import torch.nn as nn

from Common.Trainer import TrainerOneStage
from Common.Utils import DecodedBuffer, calculate_bpp, cal_psnr
from DCVC import InterFrameCodecDCVC


class TrainerDCVC(TrainerOneStage):
    def __init__(self, inter_frame_codec: nn.Module, num_available_frames: int = 2) -> None:
        super().__init__(inter_frame_codec=inter_frame_codec, num_available_frames=num_available_frames)

    def encode_sequence(self, frames: torch.Tensor, *args, **kwargs) -> dict:
        decode_buffer = DecodedBuffer()

        num_available_frames = 2

        # I frame coding
        intra_frame = frames[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")
        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]
        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = enc_results["x_hat"]
        decode_buffer.update(intra_frame_hat)

        intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
        intra_psnr = cal_psnr(intra_dist)
        intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.args.gpu else "cpu") for i in range(1, len(frames))]

        rd_cost_avg = aux_loss_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        for frame in inter_frames:
            ref = decode_buffer.get_frames(num_frames=1)

            frame_hat, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

            recon_dist = self.distortion_metric(frame_hat, frame)
            recon_psnr = cal_psnr(recon_dist)

            motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
            frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)

            rd_cost = self.args.lambda_weight * recon_dist + frame_bpp + motion_bpp
            aux_loss = self.inter_frame_codec.aux_loss()

            rd_cost_avg += rd_cost / len(inter_frames)
            aux_loss_avg += aux_loss / len(inter_frames)
            recon_psnr_avg_inter += recon_psnr / len(inter_frames)
            frame_bpp_avg += frame_bpp / len(inter_frames)
            motion_bpp_avg += motion_bpp / len(inter_frames)

            decode_buffer.update(frame_hat)

        recon_psnr_avg = (recon_psnr_avg_inter * len(inter_frames) + intra_psnr) / (len(frames))

        total_bpp_avg_inter = motion_bpp_avg + frame_bpp_avg
        total_bpp_avg = (total_bpp_avg_inter * len(inter_frames) + intra_bpp) / (len(frames) + 1)

        return {
            "rd_cost": rd_cost_avg,
            "aux_loss": aux_loss_avg,
            "recon_psnr_inter": recon_psnr_avg_inter, "recon_psnr": recon_psnr_avg,
            "motion_bpp": motion_bpp_avg, "frame_bpp": frame_bpp_avg,
            "total_bpp_inter": total_bpp_avg_inter, "total_bpp": total_bpp_avg,
            "reconstruction": decode_buffer.get_frames(len(decode_buffer)),
            "pristine": frames
        }

    def visualize(self, enc_results: dict) -> None:
        self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                     tag_scalar_dict={"Reconstruction": enc_results["recon_psnr_inter"]})
        self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                     tag_scalar_dict={"Motion Info": enc_results["motion_bpp"], "Frame": enc_results["frame_bpp"]})
        if self.train_steps % 100 == 0:
            self.tensorboard.add_images(tag="Training/Reconstruction", global_step=self.train_steps,
                                        img_tensor=torch.stack(enc_results["reconstruction"], dim=1)[0].clone().detach().cpu())
            self.tensorboard.add_images(tag="Training/Pristine", global_step=self.train_steps,
                                        img_tensor=torch.stack(enc_results["pristine"], dim=1)[0].clone().detach().cpu())


if __name__ == "__main__":
    trainer = TrainerDCVC(inter_frame_codec=InterFrameCodecDCVC)
    trainer.train()
