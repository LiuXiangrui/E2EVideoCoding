#! /usr/bin/env python3

from enum import Enum, unique

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from Model.Common.Trainer import TrainerABC
from Model.Common.Utils import calculate_bpp, cal_psnr, separate_aux_and_normal_params
from Model.DVC import InterFrameCodecDVC


@unique
class TrainingStage(Enum):
    WITH_INTER_LOSS = 0
    ONLY_RD_LOSS = 1
    NOT_AVAILABLE = 2


class TrainerDVC(TrainerABC):
    def __init__(self) -> None:
        super().__init__()
        self.inter_frame_codec = InterFrameCodecDVC(network_config=self.network_args.serialize())

    def encode_sequence(self, frames: torch.Tensor, stage: TrainingStage) -> dict:
        assert stage != TrainingStage.NOT_AVAILABLE

        frame, ref = torch.chunk(frames, chunks=2, dim=1)

        frame_hat, aligned_ref, pred, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

        align_dist = self.distortion_metric(aligned_ref, frame)
        align_psnr = cal_psnr(align_dist)

        pred_dist = self.distortion_metric(pred, frame)
        pred_psnr = cal_psnr(pred_dist)

        recon_dist = self.distortion_metric(frame_hat, frame)
        recon_psnr = cal_psnr(recon_dist)

        distortion = int(stage == TrainingStage.WITH_INTER_LOSS) * 0.1 * (align_dist + pred_dist) + recon_dist

        num_pixels = frame.shape[0] * frame.shape[2] * frame.shape[3]
        motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
        frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)
        rate = frame_bpp + motion_bpp

        rd_cost = self.training_args.lambda_weight * distortion + rate

        aux_loss = self.inter_frame_codec.aux_loss()

        return {
            "aux_loss": aux_loss,
            "rd_cost": rd_cost,
            "align_psnr": align_psnr,
            "pred_psnr": pred_psnr,
            "recon_psnr": recon_psnr,
            "motion_bpp": motion_bpp,
            "frame_bpp": frame_bpp,
            "total_bpp": rate
        }

    def visualize(self, enc_results: dict, stage: TrainingStage) -> None:
        self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                     tag_scalar_dict={"Reconstruction": enc_results["recon_psnr"],
                                                      "Prediction": enc_results["pred_psnr"],
                                                      "Alignment": enc_results["align_psnr"]})
        self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                     tag_scalar_dict={"Motion": enc_results["motion_bpp"],
                                                      "Frame": enc_results["frame_bpp"],
                                                      "Total": enc_results["total_bpp"]})

    def lr_decay(self, stage: TrainingStage) -> None:
        self.schedulers[stage].step()
        self.aux_schedulers[stage].step()

    def infer_stage(self, epoch: int) -> TrainingStage:
        epoch_milestone = self.training_args.epoch_milestone
        assert len(epoch_milestone) == TrainingStage.NOT_AVAILABLE.value
        epoch_interval = [sum(epoch_milestone[:i]) - epoch > 0 for i in range(1, len(epoch_milestone) + 1)]
        stage = TrainingStage(epoch_interval.index(True))
        return stage

    def init_optimizer(self) -> tuple:
        lr_milestone = self.training_args.lr_milestone
        assert len(lr_milestone) == 1

        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizer = Adam([{'params': params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])
        aux_optimizer = Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])

        optimizers = {
            TrainingStage.WITH_INTER_LOSS: optimizer,
            TrainingStage.ONLY_RD_LOSS: optimizer,
        }

        aux_optimizers = {
            TrainingStage.WITH_INTER_LOSS: aux_optimizer,
            TrainingStage.ONLY_RD_LOSS: aux_optimizer,
        }
        return optimizers, aux_optimizers

    def init_schedulers(self, start_epoch: int) -> tuple:
        lr_decay_milestone = self.training_args.lr_decay_milestone if isinstance(self.training_args.lr_decay_milestone, list) else [self.training_args.lr_decay_milestone, ]

        scheduler = MultiStepLR(optimizer=self.optimizers[TrainingStage.WITH_INTER_LOSS], last_epoch=start_epoch - 1,
                                milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizers[TrainingStage.WITH_INTER_LOSS], last_epoch=start_epoch - 1,
                                    milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor)

        schedulers = {
            TrainingStage.WITH_INTER_LOSS: scheduler,
            TrainingStage.ONLY_RD_LOSS: scheduler,
        }

        aux_schedulers = {
            TrainingStage.WITH_INTER_LOSS: aux_scheduler,
            TrainingStage.ONLY_RD_LOSS: aux_scheduler,
        }

        return schedulers, aux_schedulers


if __name__ == "__main__":
    trainer = TrainerDVC()
    trainer.train()
