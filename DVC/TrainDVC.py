import sys
from enum import Enum, unique

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from DVC import InterFrameCodecDVC

sys.path.append("../Common/")

from Common.Trainer import TrainerABC
from Common.Utils import DecodedFrameBuffer, calculate_bpp, cal_psnr, separate_aux_and_normal_params


@unique
class TrainingStage(Enum):
    WITH_INTER_LOSS = 0
    ONLY_RD_LOSS = 1
    ROLLING = 2
    NOT_AVAILABLE = 3


class TrainerDVC(TrainerABC):
    def __init__(self) -> None:
        super().__init__()
        self.inter_frame_codec = InterFrameCodecDVC(N_motion=self.args.N_motion, M_motion=self.args.M_motion, N_residues=self.args.N_residues, M_residues=self.args.M_residues)

    def encode_sequence(self, frames: torch.Tensor, stage: TrainingStage) -> dict:
        decode_frame_buffer = DecodedFrameBuffer()

        num_available_frames = 7 if stage == TrainingStage.ROLLING else 2
        frames = frames[:, :num_available_frames, :, :, :]

        # I frame coding
        intra_frame = frames[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")
        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]
        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = torch.clamp(enc_results["x_hat"], min=0.0, max=1.0)
        decode_frame_buffer.update(intra_frame_hat)

        intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
        intra_psnr = cal_psnr(intra_dist)
        intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.args.gpu else "cpu") for i in range(1, num_available_frames)]

        prediction = [torch.zeros_like(intra_frame_hat), ]

        rd_cost_avg = aux_loss_avg = align_psnr_avg = pred_psnr_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        for frame in inter_frames:
            ref = decode_frame_buffer.get_frames(num_frames=1)[0]

            frame_hat, aligned_ref, pred, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

            align_dist = self.distortion_metric(aligned_ref, frame)
            align_psnr = cal_psnr(align_dist)

            pred_dist = self.distortion_metric(pred, frame)
            pred_psnr = cal_psnr(pred_dist)

            recon_dist = self.distortion_metric(frame_hat, frame)
            recon_psnr = cal_psnr(recon_dist)

            motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
            frame_bpp = calculate_bpp(frame_likelihoods, num_pixels=num_pixels)

            rate = frame_bpp + motion_bpp
            distortion = 0.1 * (align_dist + pred_dist) + recon_dist if stage == TrainingStage.WITH_INTER_LOSS else recon_dist

            rd_cost = self.args.lambda_weight * distortion + rate

            aux_loss = self.inter_frame_codec.aux_loss()

            rd_cost_avg += rd_cost / len(inter_frames)
            aux_loss_avg += aux_loss / len(inter_frames)

            recon_psnr_avg_inter += recon_psnr / len(inter_frames)
            align_psnr_avg += align_psnr / len(inter_frames)
            pred_psnr_avg += pred_psnr / len(inter_frames)

            frame_bpp_avg += frame_bpp / len(inter_frames)
            motion_bpp_avg += motion_bpp / len(inter_frames)

            decode_frame_buffer.update(frame_hat)

            prediction.append(pred)

        recon_psnr_avg = (recon_psnr_avg_inter * len(inter_frames) + intra_psnr) / num_available_frames

        total_bpp_avg_inter = motion_bpp_avg + frame_bpp_avg
        total_bpp_avg = (total_bpp_avg_inter * len(inter_frames) + intra_bpp) / num_available_frames

        return {
            "rd_cost": rd_cost_avg,
            "aux_loss": aux_loss_avg,
            "align_psnr": align_psnr_avg,
            "pred_psnr": pred_psnr_avg,
            "recon_psnr_inter": recon_psnr_avg_inter,
            "recon_psnr": recon_psnr_avg,
            "motion_bpp": motion_bpp_avg, "frame_bpp": frame_bpp_avg,
            "total_bpp_inter": total_bpp_avg_inter, "total_bpp": total_bpp_avg,
            "reconstruction": decode_frame_buffer.get_frames(num_frames=num_available_frames),
            "pristine": [frames[:, i, :, :, :] for i in range(num_available_frames)],
            "prediction": prediction
        }

    def visualize(self, enc_results: dict, stage: TrainingStage) -> None:
        self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                     tag_scalar_dict={"Reconstruction": enc_results["recon_psnr_inter"],
                                                      "Prediction": enc_results["pred_psnr"],
                                                      "Alignment": enc_results["align_psnr"]})
        self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                     tag_scalar_dict={"Motion Info": enc_results["motion_bpp"], "Frame": enc_results["frame_bpp"]})
        if self.train_steps % 2000 == 0:
            assert len(enc_results["pristine"]) == len(enc_results["reconstruction"]) and len(enc_results["pristine"]) == len(enc_results["prediction"])

            for i in range(len(enc_results["pristine"])):
                self.tensorboard.add_images(tag="Training/Reconstruction_Frame_{}".format(str(i + 1)), global_step=self.train_steps,
                                            img_tensor=enc_results["reconstruction"][i].clone().detach().cpu())
                self.tensorboard.add_images(tag="Training/Pristine_Frame_{}".format(str(i + 1)), global_step=self.train_steps,
                                            img_tensor=enc_results["pristine"][i].clone().detach().cpu())
                self.tensorboard.add_images(tag="Training/Prediction_Frame_{}".format(str(i + 1)), global_step=self.train_steps,
                                            img_tensor=enc_results["prediction"][i].clone().detach().cpu())

    def lr_decay(self, stage: TrainingStage) -> None:
        self.schedulers[stage].step()
        self.aux_schedulers[stage].step()

    def infer_stage(self, epoch: int) -> TrainingStage:
        epoch_milestone = self.args.epoch_milestone if isinstance(self.args.epoch_milestone, list) else [self.args.epoch_milestone, ]
        assert len(epoch_milestone) == TrainingStage.NOT_AVAILABLE.value
        epoch_interval = [sum(epoch_milestone[:i]) - epoch > 0 for i in range(1, len(epoch_milestone) + 1)]
        stage = TrainingStage(epoch_interval.index(True))
        return stage

    def init_optimizer(self) -> tuple[dict, dict]:
        lr_milestone = self.args.lr_milestone if isinstance(self.args.lr_milestone, list) else [self.args.lr_milestone, ]
        assert len(lr_milestone) == 1

        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizer = Adam([{'params': params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])
        aux_optimizer = Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])

        optimizers = {
            TrainingStage.WITH_INTER_LOSS: optimizer,
            TrainingStage.ONLY_RD_LOSS: optimizer,
            TrainingStage.ROLLING: optimizer
        }

        aux_optimizers = {
            TrainingStage.WITH_INTER_LOSS: aux_optimizer,
            TrainingStage.ONLY_RD_LOSS: aux_optimizer,
            TrainingStage.ROLLING: aux_optimizer
        }
        return optimizers, aux_optimizers

    def init_schedulers(self, start_epoch: int) -> tuple[dict, dict]:
        lr_decay_milestone = self.args.lr_decay_milestone if isinstance(self.args.lr_decay_milestone, list) else [self.args.lr_decay_milestone, ]

        scheduler = MultiStepLR(optimizer=self.optimizers[TrainingStage.WITH_INTER_LOSS], last_epoch=start_epoch - 1,
                                milestones=lr_decay_milestone, gamma=self.args.lr_decay_factor)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizers[TrainingStage.WITH_INTER_LOSS], last_epoch=start_epoch - 1,
                                    milestones=lr_decay_milestone, gamma=self.args.lr_decay_factor)

        schedulers = {
            TrainingStage.WITH_INTER_LOSS: scheduler,
            TrainingStage.ONLY_RD_LOSS: scheduler,
            TrainingStage.ROLLING: scheduler
        }

        aux_schedulers = {
            TrainingStage.WITH_INTER_LOSS: aux_scheduler,
            TrainingStage.ONLY_RD_LOSS: aux_scheduler,
            TrainingStage.ROLLING: aux_scheduler
        }

        return schedulers, aux_schedulers


if __name__ == "__main__":
    trainer = TrainerDVC()
    trainer.train()
