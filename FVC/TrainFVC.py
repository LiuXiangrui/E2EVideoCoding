from enum import Enum, unique

import torch
import torch.nn as nn
from torch.optim import Adam

from Common.Trainer import Trainer
from Common.Utils import DecodedFrameBuffer, calculate_bpp, cal_psnr, separate_aux_and_normal_params
from FVC import InterFrameCodecFVC


@unique
class TrainingStage(Enum):
    WITHOUT_FUSION = 1,
    WITH_FUSION = 2,
    FINE_TUNE = 3,
    NOT_AVAILABLE = 4


class TrainerFVC(Trainer):
    def __init__(self, inter_frame_codec: nn.Module) -> None:
        super().__init__(inter_frame_codec=inter_frame_codec)

    def get_stage(self, epoch: int) -> TrainingStage:
        epoch_milestone = self.args.epoch_milestone
        if epoch < epoch_milestone[:1]:
            stage = TrainingStage.WITHOUT_FUSION
        elif epoch < sum(epoch_milestone[:2]):
            stage = TrainingStage.WITH_FUSION
        elif epoch < sum(epoch_milestone[:3]):
            stage = TrainingStage.FINE_TUNE
        else:
            stage = TrainingStage.NOT_AVAILABLE
        return stage

    def train(self):
        start_epoch, best_rd_cost = self.load_checkpoints()

        epoch_milestone = self.args.epoch_milestone
        assert len(epoch_milestone) == 3

        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            print("\nEpoch {0}, stage is '{1}}'".format(str(epoch), self.get_stage(epoch)))
            self.train_one_epoch(stage=self.get_stage(epoch))

            if epoch % self.args.eval_epochs == 0:
                rd_cost = self.evaluate(stage=self.get_stage(epoch))
                if epoch % self.args.save_epochs == 0 or rd_cost < best_rd_cost:
                    best_rd_cost = min(best_rd_cost, rd_cost)
                    self.save_ckpt(epoch=epoch, best_rd_cost=best_rd_cost, stage=self.get_stage(epoch))

    def encode_sequence(self, frames: torch.Tensor, stage: TrainingStage = TrainingStage.NOT_AVAILABLE, *args, **kwargs) -> dict:
        decode_frame_buffer = DecodedFrameBuffer()

        num_available_frames = 2 if stage == TrainingStage.WITHOUT_FUSION else 4
        frames = frames[:, :num_available_frames, :, :, :]

        # I frame coding
        intra_frame = frames[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")
        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]
        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = enc_results["x_hat"]

        decode_frame_buffer.update(intra_frame_hat)

        intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
        intra_psnr = cal_psnr(intra_dist)
        intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.args.gpu else "cpu") for i in range(1, len(frames))]

        rd_cost_avg = aux_loss_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        for frame in inter_frames:
            ref_frames_list = decode_frame_buffer.get_frames(num_frames=1 if stage == TrainingStage.WITHOUT_FUSION else max(len(decode_frame_buffer), 3))

            frame_hat, feats_hat, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref_frames_list=ref_frames_list, fusion=(stage != TrainingStage.WITHOUT_FUSION))

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

            decode_frame_buffer.update(frame_hat)

        recon_psnr_avg = (recon_psnr_avg_inter * len(inter_frames) + intra_psnr) / (len(frames))

        total_bpp_avg_inter = motion_bpp_avg + frame_bpp_avg
        total_bpp_avg = (total_bpp_avg_inter * len(inter_frames) + intra_bpp) / (len(frames) + 1)

        return {
            "rd_cost": rd_cost_avg,
            "aux_loss": aux_loss_avg,
            "recon_psnr_inter": recon_psnr_avg_inter, "recon_psnr": recon_psnr_avg,
            "motion_bpp": motion_bpp_avg, "frame_bpp": frame_bpp_avg,
            "total_bpp_inter": total_bpp_avg_inter, "total_bpp": total_bpp_avg,
            "reconstruction": decode_frame_buffer.get_frames(len(decode_frame_buffer)),
            "pristine": frames
        }

    def optimize(self, enc_results: dict, stage: TrainingStage = TrainingStage.NOT_AVAILABLE, *args, **kwargs) -> None:
        optimizer = self.optimizers[stage]
        loss = enc_results["rd_cost"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        optimizer.step()
        optimizer.zero_grad()

        aux_optimizer = self.aux_optimizers[stage]
        loss = enc_results["aux_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        aux_optimizer.step()
        aux_optimizer.zero_grad()

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

    def lr_decay(self, *args, **kwargs) -> None:
        pass

    def init_optimizer(self) -> tuple:
        lr_milestone = self.args.lr_milestone
        assert len(lr_milestone) == 3

        params_w_o_fusion, _ = separate_aux_and_normal_params(self.inter_frame_codec, exclude_net=self.inter_frame_codec.post_processing)
        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizers = {
            TrainingStage.WITHOUT_FUSION:  Adam([{'params': params_w_o_fusion, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0]),
            TrainingStage.WITH_FUSIONW: Adam([{'params': params, 'initial_lr': lr_milestone[1]}], lr=lr_milestone[1]),
            TrainingStage.FINE_TUNE: Adam([{'params': params, 'initial_lr': lr_milestone[2]}], lr=lr_milestone[2])
        }

        aux_optimizers = {
            TrainingStage.WITHOUT_FUSION: Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0]),
            TrainingStage.WITH_FUSIONW: Adam([{'params': aux_params, 'initial_lr': lr_milestone[1]}], lr=lr_milestone[1]),
            TrainingStage.FINE_TUNE: Adam([{'params': aux_params, 'initial_lr': lr_milestone[2]}], lr=lr_milestone[2]),
        }
        return optimizers, aux_optimizers

    def init_schedulers(self, start_epoch: int) -> tuple:
        pass

    def load_checkpoints(self) -> tuple:
        start_epoch = 0
        best_rd_cost = 1e9
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")

            best_rd_cost = ckpt['best_rd_cost']

            start_epoch = ckpt["epoch"] + 1

            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])

            stage = self.get_stage(epoch=ckpt["epoch"])
            self.optimizers[stage].load_state_dict(ckpt["optimizer"])
            self.aux_optimizers[stage].load_state_dict(ckpt["aux_optimizer"])

        elif self.args.pretrained:
            ckpt = torch.load(self.args.pretrained)
            print("\n===========Load pretrained {0}===========\n".format(self.args.pretrained))
            pretrained_dict = ckpt["inter_frame_codec"]
            model_dict = self.inter_frame_codec.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.inter_frame_codec.load_state_dict(model_dict)
        else:
            print("\n===========Training from scratch===========\n")
        return start_epoch, best_rd_cost

    def save_ckpt(self, epoch: int, best_rd_cost: torch.Tensor, stage: TrainingStage = TrainingStage.NOT_AVAILABLE, *args, **kwargs) -> None:
        ckpt = {
            "inter_frame_codec": self.inter_frame_codec.state_dict(),
            "optimizer": self.optimizers[stage].state_dict(),
            "aux_optimizer": self.aux_optimizers[stage].state_dict(),
            "epoch": epoch,
            "best_rd_cost": best_rd_cost
        }
        ckpt_path = "%s/DCVC_Inter_%.3d.pth" % (self.checkpoints_dir, epoch)
        torch.save(ckpt, ckpt_path)
        print("\nSave model to " + ckpt_path)


if __name__ == "__main__":
    trainer = Trainer(inter_frame_codec=InterFrameCodecFVC)
    trainer.train()
