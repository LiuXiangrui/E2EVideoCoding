from enum import Enum, unique

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from Model.Common.Trainer import TrainerABC
from Model.Common.Utils import DecodedFrameBuffer, calculate_bpp, cal_psnr, separate_aux_and_normal_params
from FVC import InterFrameCodecFVC


@unique
class TrainingStage(Enum):
    WITHOUT_FUSION = 0,
    WITH_FUSION = 1,
    NOT_AVAILABLE = 2


class TrainerFVC(TrainerABC):
    def __init__(self) -> None:
        super().__init__()
        self.inter_frame_codec = InterFrameCodecFVC(N=self.training_args.N, M=self.training_args.M)

    def encode_sequence(self, frames: torch.Tensor, stage: TrainingStage) -> dict:
        assert stage != TrainingStage.NOT_AVAILABLE

        decode_frame_buffer = DecodedFrameBuffer()

        num_available_frames = 2 if stage == TrainingStage.WITHOUT_FUSION else 4
        frames = frames[:, :num_available_frames, :, :, :]

        # I frame coding
        intra_frame = frames[:, 0, :, :, :].to("cuda" if self.training_args.gpu else "cpu")
        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]
        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = torch.clamp(enc_results["x_hat"], min=0.0, max=1.0)

        decode_frame_buffer.update(intra_frame_hat)

        intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
        intra_psnr = cal_psnr(intra_dist)
        intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.training_args.gpu else "cpu") for i in range(1, num_available_frames)]

        rd_cost_avg = aux_loss_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        for frame in inter_frames:
            ref_frames_list = decode_frame_buffer.get_frames(num_frames=1 if stage == TrainingStage.WITHOUT_FUSION else min(len(decode_frame_buffer), 3))

            frame_hat, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref_frames_list=ref_frames_list, fusion=(stage != TrainingStage.WITHOUT_FUSION))

            recon_dist = self.distortion_metric(frame_hat, frame)
            recon_psnr = cal_psnr(recon_dist)

            motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
            frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)

            rd_cost = self.training_args.lambda_weight * recon_dist + frame_bpp + motion_bpp
            aux_loss = self.inter_frame_codec.aux_loss()

            rd_cost_avg += rd_cost / len(inter_frames)
            aux_loss_avg += aux_loss / len(inter_frames)
            recon_psnr_avg_inter += recon_psnr / len(inter_frames)
            frame_bpp_avg += frame_bpp / len(inter_frames)
            motion_bpp_avg += motion_bpp / len(inter_frames)

            decode_frame_buffer.update(frame_hat)

        recon_psnr_avg = (recon_psnr_avg_inter * len(inter_frames) + intra_psnr) / num_available_frames

        total_bpp_avg_inter = motion_bpp_avg + frame_bpp_avg
        total_bpp_avg = (total_bpp_avg_inter * len(inter_frames) + intra_bpp) / num_available_frames

        return {
            "rd_cost": rd_cost_avg,
            "aux_loss": aux_loss_avg,
            "recon_psnr_inter": recon_psnr_avg_inter, "recon_psnr": recon_psnr_avg,
            "motion_bpp": motion_bpp_avg, "frame_bpp": frame_bpp_avg,
            "total_bpp_inter": total_bpp_avg_inter, "total_bpp": total_bpp_avg,
            "reconstruction": decode_frame_buffer.get_frames(len(decode_frame_buffer)),
            "pristine": [frames[:, i, :, :, :] for i in range(num_available_frames)]
        }

    def visualize(self, enc_results: dict, stage: TrainingStage) -> None:
        self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                     tag_scalar_dict={"Reconstruction": enc_results["recon_psnr_inter"]})
        self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                     tag_scalar_dict={"Motion Info": enc_results["motion_bpp"], "Frame": enc_results["frame_bpp"]})
        if self.train_steps % 1000 == 0:
            for i in range(len(enc_results["pristine"])):
                self.tensorboard.add_images(tag="Training/Reconstruction_Frame_{}".format(str(i + 1)), global_step=self.train_steps,
                                            img_tensor=enc_results["reconstruction"][i].clone().detach().cpu())
                self.tensorboard.add_images(tag="Training/Pristine_Frame_{}".format(str(i + 1)), global_step=self.train_steps,
                                            img_tensor=enc_results["pristine"][i].clone().detach().cpu())

    def infer_stage(self, epoch: int) -> TrainingStage:
        epoch_milestone = self.training_args.epoch_milestone if isinstance(self.training_args.epoch_milestone, list) else [self.training_args.epoch_milestone, ]
        assert len(epoch_milestone) == TrainingStage.NOT_AVAILABLE.value
        epoch_interval = [sum(epoch_milestone[:i]) - epoch > 0 for i in range(1, len(epoch_milestone) + 1)]
        stage = TrainingStage(epoch_interval.index(True))
        return stage

    def init_optimizer(self) -> tuple[dict, dict]:
        lr_milestone = self.training_args.lr_milestone
        assert len(lr_milestone) == 2

        params_w_o_fusion, _ = separate_aux_and_normal_params(self.inter_frame_codec, exclude_module_list=self.inter_frame_codec.post_processing)
        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizers = {
            TrainingStage.WITHOUT_FUSION:  Adam([{'params': params_w_o_fusion, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0]),
            TrainingStage.WITH_FUSION: Adam([{'params': params, 'initial_lr': lr_milestone[1]}], lr=lr_milestone[1]),
        }

        aux_optimizers = {
            TrainingStage.WITHOUT_FUSION: Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0]),
            TrainingStage.WITH_FUSION: Adam([{'params': aux_params, 'initial_lr': lr_milestone[1]}], lr=lr_milestone[1]),
        }

        return optimizers, aux_optimizers

    def init_schedulers(self, start_epoch: int) -> tuple[dict, dict]:
        lr_decay_milestone = self.training_args.lr_decay_milestone

        schedulers = {
            TrainingStage.WITHOUT_FUSION: MultiStepLR(optimizer=self.optimizers[TrainingStage.WITHOUT_FUSION], last_epoch=start_epoch - 1,
                                                      milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor),
            TrainingStage.WITH_FUSION: MultiStepLR(optimizer=self.optimizers[TrainingStage.WITH_FUSION], last_epoch=start_epoch - 1,
                                                   milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor),
        }

        aux_schedulers = {
            TrainingStage.WITHOUT_FUSION: MultiStepLR(optimizer=self.aux_optimizers[TrainingStage.WITHOUT_FUSION], last_epoch=start_epoch - 1,
                                                      milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor),
            TrainingStage.WITH_FUSION: MultiStepLR(optimizer=self.aux_optimizers[TrainingStage.WITH_FUSION], last_epoch=start_epoch - 1,
                                                   milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor)
        }

        return schedulers, aux_schedulers


if __name__ == "__main__":
    trainer = TrainerFVC()
    trainer.train()
