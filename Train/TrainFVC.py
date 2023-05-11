#! /usr/bin/env python3

from enum import Enum, unique

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop

from Model.Common.Dataset import Vimeo90KDataset
from Model.Common.Trainer import TrainerABC
from Model.Common.Utils import DecodedFrameBuffer, calculate_bpp, cal_psnr, separate_aux_and_normal_params
from Model.FVC import InterFrameCodecFVC


@unique
class TrainingStage(Enum):
    ONLY_RD_LOSS = 0
    ROLLING = 1
    NOT_AVAILABLE = 2


class TrainerFVC(TrainerABC):
    def __init__(self) -> None:
        super().__init__()
        self.inter_frame_codec = InterFrameCodecFVC(network_config=self.network_args.serialize())

    def encode_sequence(self, frames: torch.Tensor, stage: TrainingStage) -> dict:
        assert stage != TrainingStage.NOT_AVAILABLE

        decode_frame_buffer = DecodedFrameBuffer(capacity=self.training_args.decode_buffer_capacity)

        num_available_frames = 7 if stage == TrainingStage.ROLLING else 2
        frames = frames[:, :num_available_frames, :, :, :]

        # I frame coding
        intra_frame = frames[0].to("cuda" if self.training_args.gpu else "cpu")
        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = torch.clamp(enc_results["x_hat"], min=0.0, max=1.0)
        decode_frame_buffer.update(intra_frame_hat)

        rd_cost_avg = aux_loss_avg = recon_psnr_avg = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        inter_frames = frames[1:]
        for frame in inter_frames:
            frame = frame.to("cuda" if self.training_args.gpu else "cpu")
            ref = decode_frame_buffer.get_frames(num_frames=1)[0].to("cuda" if self.training_args.gpu else "cpu")

            frame_hat, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

            # calculate distortion
            recon_dist = self.distortion_metric(frame_hat, frame)
            distortion = recon_dist

            # calculate rate
            num_pixels = frame.shape[0] * frame.shape[2] * frame.shape[3]
            motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
            frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)
            rate = frame_bpp + motion_bpp

            # calculate loss
            rd_cost = self.training_args.lambda_weight * distortion + rate
            aux_loss = self.inter_frame_codec.aux_loss()

            # calculate PSNR
            recon_psnr = cal_psnr(recon_dist)

            # update record
            rd_cost_avg += rd_cost / len(inter_frames)
            aux_loss_avg += aux_loss / len(inter_frames)

            recon_psnr_avg += recon_psnr / len(inter_frames)

            frame_bpp_avg += frame_bpp / len(inter_frames)
            motion_bpp_avg += motion_bpp / len(inter_frames)

            decode_frame_buffer.update(frame_hat)

        return {
            "rd_cost": rd_cost_avg,
            "aux_loss": aux_loss_avg,
            "recon_psnr": recon_psnr_avg,
            "motion_bpp": motion_bpp_avg,
            "frame_bpp": frame_bpp_avg,
            "total_bpp": frame_bpp_avg + motion_bpp_avg
        }

    def visualize(self, enc_results: dict, stage: TrainingStage) -> None:
        self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                     tag_scalar_dict={"Reconstruction": enc_results["recon_psnr"]})
        self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                     tag_scalar_dict={"Motion": enc_results["motion_bpp"],
                                                      "Frame": enc_results["frame_bpp"],
                                                      "Total": enc_results["total_bpp"]})

    def lr_decay(self, stage: TrainingStage) -> None:
        self.schedulers[stage].step()
        self.aux_schedulers[stage].step()

    def infer_stage(self, epoch: int) -> TrainingStage:
        epoch_milestone = self.training_args.epoch_milestone if isinstance(self.training_args.epoch_milestone, list) else [self.training_args.epoch_milestone, ]
        assert len(epoch_milestone) == TrainingStage.NOT_AVAILABLE.value
        epoch_interval = [sum(epoch_milestone[:i]) - epoch > 0 for i in range(1, len(epoch_milestone) + 1)]
        stage = TrainingStage(epoch_interval.index(True))
        return stage

    def init_optimizer(self) -> tuple[dict, dict]:
        lr_milestone = self.training_args.lr_milestone
        assert len(lr_milestone) == 1

        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizer = Adam([{'params': params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])
        aux_optimizer = Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])

        optimizers = {
            TrainingStage.ONLY_RD_LOSS: optimizer,
            TrainingStage.ROLLING: optimizer
        }

        aux_optimizers = {
            TrainingStage.ONLY_RD_LOSS: aux_optimizer,
            TrainingStage.ROLLING: aux_optimizer
        }

        return optimizers, aux_optimizers

    def init_schedulers(self, start_epoch: int) -> tuple:
        lr_decay_milestone = self.training_args.lr_decay_milestone if isinstance(self.training_args.lr_decay_milestone, list) else [self.training_args.lr_decay_milestone, ]

        scheduler = MultiStepLR(optimizer=self.optimizers[TrainingStage.ONLY_RD_LOSS], last_epoch=start_epoch - 1,
                                milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizers[TrainingStage.ONLY_RD_LOSS],
                                    last_epoch=start_epoch - 1,
                                    milestones=lr_decay_milestone, gamma=self.training_args.lr_decay_factor)

        schedulers = {
            TrainingStage.ONLY_RD_LOSS: scheduler,
            TrainingStage.ROLLING: scheduler
        }

        aux_schedulers = {
            TrainingStage.ONLY_RD_LOSS: aux_scheduler,
            TrainingStage.ROLLING: scheduler
        }

        return schedulers, aux_schedulers

    def init_dataloader(self) -> tuple:
        train_dataloader_single = DataLoader(  # use single P frame
            dataset=Vimeo90KDataset(
                root=self.training_args.dataset_root, training_frames_list_path=self.training_args.split_filepath,
                transform=Compose([RandomCrop(size=256), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)])),
            batch_size=self.training_args.batch_size, shuffle=True, pin_memory=True,
            num_workers=self.training_args.num_workers)

        train_dataloader_rolling = DataLoader(  # use multiple P frames
            dataset=Vimeo90KDataset(
                root=self.training_args.dataset_root, training_frames_list_path=self.training_args.split_filepath,
                transform=Compose([RandomCrop(size=256), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)]),
                use_all_frames=True),
            batch_size=self.training_args.batch_size, shuffle=True, pin_memory=True,
            num_workers=self.training_args.num_workers)

        train_dataloaders = {
            TrainingStage.ONLY_RD_LOSS: train_dataloader_single,
            TrainingStage.ROLLING: train_dataloader_rolling
        }

        eval_dataloader = None

        return train_dataloaders, eval_dataloader


if __name__ == "__main__":
    trainer = TrainerFVC()
    trainer.train()
