import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Common.Trainer import Trainer
from Common.Utils import DecodedBuffer, calculate_bpp, cal_psnr, separate_aux_and_normal_params
from DCVC import InterFrameCodecDCVC


class TrainerDCVC(Trainer):
    def __init__(self, inter_frame_codec: nn.Module):
        super().__init__(inter_frame_codec=inter_frame_codec)

    def init_optimizer(self) -> tuple:
        lr_milestone = self.args.lr_milestone
        assert len(lr_milestone) == 1
        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizer = Adam([{'params': params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])
        aux_optimizer = Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])

        return [optimizer, ], [aux_optimizer, ]

    def encode_sequence(self, frames: torch.Tensor) -> dict:
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

        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.args.gpu else "cpu") for i in
                        range(1, num_available_frames)]

        rd_cost_avg = aux_loss_avg = recon_psnr_avg_inter = motion_bpp_avg = frame_bpp_avg = 0.

        # P frame coding
        for frame in inter_frames:
            ref = decode_buffer.get_frames(num_frames=1)

            frame_hat, _, motion_likelihoods, frame_likelihoods = self.inter_frame_codec(frame, ref=ref)

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
                "rd_cost_avg": rd_cost_avg,
                "aux_loss_avg": aux_loss_avg,
                "recon_psnr_avg_inter": recon_psnr_avg_inter, "recon_psnr_avg": recon_psnr_avg,
                "motion_bpp_avg": motion_bpp_avg, "frame_bpp_avg": frame_bpp_avg,
                "total_bpp_avg_inter": total_bpp_avg_inter, "total_bpp_avg": total_bpp_avg,
                "reconstruction": decode_buffer.get_frames(len(decode_buffer)),
                "pristine": frames
            }

    def train_one_epoch(self):
        self.inter_frame_codec.train()
        for sequence in tqdm(self.train_dataloader, total=len(self.train_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode_sequence(sequence)

            optimizer = self.optimizers[0]
            optimizer.zero_grad()
            loss = encode_results["rd_cost_avg"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
            optimizer.step()

            aux_optimizer = self.aux_optimizers[0]
            aux_optimizer.zero_grad()
            loss = encode_results["aux_loss_avg"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
            aux_optimizer.step()

            self.train_steps += 1
            self.tensorboard.add_scalars(main_tag="Training/PSNR", global_step=self.train_steps,
                                         tag_scalar_dict={"Reconstruction": encode_results["recon_psnr_avg_inter"]})
            self.tensorboard.add_scalars(main_tag="Training/Bpp", global_step=self.train_steps,
                                         tag_scalar_dict={"Motion Info": encode_results["motion_bpp_avg"],
                                                          "Frame": encode_results["frame_bpp_avg"]})
            if self.train_steps % 100 == 0:
                self.tensorboard.add_images(tag="Training/Reconstruction", global_step=self.train_steps,
                                            img_tensor=torch.stack(encode_results["reconstruction"], dim=1)[0].clone().detach().cpu())
                self.tensorboard.add_images(tag="Training/Pristine", global_step=self.train_steps,
                                            img_tensor=torch.stack(encode_results["pristine"], dim=1)[0].clone().detach().cpu())

    def load_checkpoints(self):
        start_epoch = 0
        best_rd_cost = 1e9
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")
            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])
            try:
                self.optimizers[0].load_state_dict(ckpt["optimizer"])
                self.aux_optimizers[0].load_state_dict(ckpt["aux_optimizer"])
            except:
                print("Can not find some optimizers params, just ignore")
            try:
                best_rd_cost = ckpt['best_rd_cost']
            except:
                print("Can not find the record of the best rd cost, just set it to 1e9 as default.")
            try:
                start_epoch = ckpt["epoch"] + 1
            except:
                print("Can not find start epoch, just set to 0 as default.")

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

    def train(self):
        start_epoch, best_rd_cost = self.load_checkpoints()
        scheduler = MultiStepLR(optimizer=self.optimizers[0], milestones=self.args.lr_decay_milestone, gamma=self.args.lr_decay_factor, last_epoch=start_epoch - 1)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizers[0], milestones=self.args.lr_decay_milestone, gamma=self.args.lr_decay_factor, last_epoch=start_epoch - 1)

        epoch_milestone = self.args.epoch_milestone
        assert len(epoch_milestone) == 1
        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            print("\nEpoch {0}".format(str(epoch)))
            self.train_one_epoch()
            scheduler.step()
            aux_scheduler.step()

            if epoch % self.args.eval_epochs == 0:
                rd_cost = self.evaluate()
                if epoch % self.args.save_epochs == 0 or rd_cost < best_rd_cost:
                    best_rd_cost = min(best_rd_cost, rd_cost)
                    self.save_ckpt(epoch=epoch, best_rd_cost=best_rd_cost)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.inter_frame_codec.eval()
        for frames in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode_sequence(frames)
            self.record.update("rd_cost", encode_results["rd_cost_avg"].item())
            self.record.update("recon_psnr", encode_results["recon_psnr_avg"].item())
            self.record.update("recon_psnr_inter", encode_results["recon_psnr_avg_inter"].item())
            self.record.update("motion_bpp", encode_results["motion_bpp_avg"].item())
            self.record.update("frame_bpp_avg", encode_results["frame_bpp_avg"].item())
            self.record.update("total_bpp", encode_results["total_bpp_avg"].item())
            self.record.update("total_bpp_inter", encode_results["total_bpp_avg_inter"].item())
        rd_cost = self.record.get("rd_cost", average=True)
        info = self.record.display()
        self.logger.info(info)
        return rd_cost

    def save_ckpt(self, epoch: int, best_rd_cost: torch.Tensor):
        ckpt = {
            "inter_frame_codec": self.inter_frame_codec.state_dict(),
            "optimizer": self.optimizers[0].state_dict(),
            "aux_optimizer": self.aux_optimizers[0].state_dict(),
            "epoch": epoch,
            "best_rd_cost": best_rd_cost
        }
        ckpt_path = "%s/DCVC_Inter_%.3d.pth" % (self.checkpoints_dir, epoch)
        torch.save(ckpt, ckpt_path)
        print("\nSave model to " + ckpt_path)


if __name__ == "__main__":
    trainer = TrainerDCVC(inter_frame_codec=InterFrameCodecDCVC)
    trainer.train()
