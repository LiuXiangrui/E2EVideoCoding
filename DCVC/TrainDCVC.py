import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import itertools
from tqdm import tqdm
from torchvision.transforms import RandomCrop
from compressai.zoo import cheng2020_attn as IntraFrameCodec

from DCVC.DCVC import InterFrameCodecDCVC as InterFrameCodec
from Common.Dataset import Vimeo90KDataset
from Common.Utils import DecodedBuffer, calculate_bpp, cal_psnr, get_normal_params, Record, init


class Trainer:
    def __init__(self):
        self.args, self.logger, self.checkpoints_dir, self.tensorboard = init()
        self.record = Record(name=[
            'rd_cost',
            'recon_psnr', 'recon_psnr_inter',
            'motion_bpp', 'residues_bpp',
            'total_bpp', 'total_bpp_inter'
        ])

        self.inter_frame_codec = InterFrameCodec()
        self.intra_frame_codec = IntraFrameCodec(quality=self.args.quality, metric="mse", pretrained=True)
        self.intra_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.inter_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.intra_frame_codec.eval()

        self.optimizers = self.init_optimizer()

        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

        self.distortion_metric = nn.MSELoss()

        self.train_steps = 0

    def init_optimizer(self):
        lr_milestone = self.args.lr_milestone
        optimizers = {
            "inter_prediction": Adam(
                [{'params': itertools.chain(get_normal_params(self.inter_frame_codec.motion_compression),
                                            get_normal_params(self.inter_frame_codec.motion_comp)),
                  'initial_lr': lr_milestone[0]}], lr=lr_milestone[0]),
            "residues_compression": Adam(
                [{'params': get_normal_params(self.inter_frame_codec.residues_compression),
                  'initial_lr': lr_milestone[1]}], lr=lr_milestone[1]),

            "total": Adam(
                [{'params': get_normal_params(self.inter_frame_codec),
                 'initial_lr': lr_milestone[2]}], lr=lr_milestone[2])
        }
        return optimizers

    def init_dataloader(self):
        train_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_trainlist.txt", transform=RandomCrop(size=256))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
        eval_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_testlist.txt")
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)
        return train_dataloader, eval_dataloader

    def encode(self, frames: torch.Tensor, stage: str):
        decode_buffer = DecodedBuffer()

        num_available_frames = len(frames) if stage == "rolling_frames" else 2
        intra_frame = frames[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")
        inter_frames = [frames[:, i, :, :, :].to("cuda" if self.args.gpu else "cpu") for i in
                        range(1, num_available_frames)]

        num_pixels = intra_frame.shape[0] * intra_frame.shape[2] * intra_frame.shape[3]

        with torch.no_grad():
            enc_results = self.intra_frame_codec(intra_frame)
        intra_frame_hat = enc_results["x_hat"]
        decode_buffer.update(intra_frame_hat)

        intra_dist = self.distortion_metric(intra_frame_hat, intra_frame)
        intra_psnr = cal_psnr(intra_dist)
        intra_bpp = calculate_bpp(enc_results["likelihoods"], num_pixels=num_pixels)

        rd_cost_avg = torch.zeros(0)
        pred_psnr_avg_inter = torch.zeros(0)
        recon_psnr_avg_inter = torch.zeros(0)
        motion_bpp_avg = torch.zeros(0)
        residues_bpp_avg = torch.zeros(0)

        for frame in inter_frames:
            ref = decode_buffer.get_frames(num_frames=1)

            if stage is "inter_prediction":
                pred, motion_likelihoods = self.inter_frame_codec.inter_predict(frame, ref=ref)

                pred_dist = self.distortion_metric(pred, frame)
                pred_psnr = cal_psnr(pred_dist)
                motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)

                rd_cost = self.args.lambda_weight * pred_dist + motion_bpp
                rd_cost_avg += rd_cost
                pred_psnr_avg_inter += pred_psnr
                motion_bpp_avg += motion_bpp

            elif stage is "residues_compression":
                with torch.no_grad():
                    pred, _ = self.inter_frame_codec.inter_predict(frame, ref=ref)
                frame_hat, residues_likelihoods = self.inter_frame_codec.residues_compression(frame, pred=pred)

                recon_dist = self.distortion_metric(frame_hat, frame)
                recon_psnr = cal_psnr(recon_dist)

                residues_bpp = calculate_bpp(residues_likelihoods, num_pixels=num_pixels)

                rd_cost = self.args.lambda_weight * recon_dist + residues_bpp

                rd_cost_avg += rd_cost
                recon_psnr_avg_inter += recon_psnr
                residues_bpp_avg += residues_bpp

                decode_buffer.update(frame_hat)

            elif stage is "total" or "rolling":
                pred, motion_likelihoods = self.inter_frame_codec.inter_predict(frame, ref=ref)
                frame_hat, residues_likelihoods = self.inter_frame_codec.residues_compression(frame, pred=pred)

                pred_dist = self.distortion_metric(pred, frame)
                pred_psnr = cal_psnr(pred_dist)

                recon_dist = self.distortion_metric(frame_hat, frame)
                recon_psnr = cal_psnr(recon_dist)

                motion_bpp = calculate_bpp(motion_likelihoods, num_pixels=num_pixels)
                residues_bpp = calculate_bpp(residues_likelihoods, num_pixels=num_pixels)

                rd_cost = self.args.lambda_weight * recon_dist + residues_bpp + motion_bpp

                rd_cost_avg += rd_cost
                recon_psnr_avg_inter += recon_psnr
                residues_bpp_avg += residues_bpp
                pred_psnr_avg_inter += pred_psnr
                motion_bpp_avg += motion_bpp

                decode_buffer.update(frame_hat)

        rd_cost_avg = rd_cost_avg / len(frames)

        pred_psnr_avg_inter = pred_psnr_avg_inter / len(frames)

        recon_psnr_avg = (recon_psnr_avg_inter + intra_psnr) / (len(frames) + 1)
        recon_psnr_avg_inter = recon_psnr_avg_inter / len(frames)

        total_bpp_avg_inter = motion_bpp_avg + residues_bpp_avg
        total_bpp_avg = (total_bpp_avg_inter + intra_bpp) / (len(frames) + 1)
        total_bpp_avg_inter = total_bpp_avg_inter / len(frames)

        motion_bpp_avg = motion_bpp_avg / len(frames)

        residues_bpp_avg = residues_bpp_avg / len(frames)

        return {
            "rd_cost_avg": rd_cost_avg,
            "pred_psnr_avg_inter": pred_psnr_avg_inter,
            "recon_psnr_avg_inter": recon_psnr_avg_inter, "recon_psnr_avg": recon_psnr_avg,
            "motion_bpp_avg": motion_bpp_avg, "residues_bpp_avg": residues_bpp_avg,
            "total_bpp_avg_inter": total_bpp_avg_inter, "total_bpp_avg": total_bpp_avg
        }

    def train_one_epoch(self, stage: str):
        self.inter_frame_codec.train()
        for frames in tqdm(self.train_dataloader, total=len(self.train_dataloader), smoothing=0.9, ncols=50):
            self.train_steps += 1
            encode_results = self.encode(frames, stage=stage)

            if stage is "rolling":
                optimizer = self.optimizers["total"]
            else:
                optimizer = self.optimizers[stage]

            optimizer.zero_grad()
            loss = torch.tensor(encode_results["rd_cost_avg"])
            loss.backward()
            nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
            optimizer.step()

            self.tensorboard.add_scalar(tag="Training/PSNR/Prediction", scalar_value=encode_results["pred_psnr_avg_inter"], global_step=self.train_steps)
            self.tensorboard.add_scalar(tag="Training/PSNR/Reconstruction", scalar_value=encode_results["recon_psnr_avg_inter"], global_step=self.train_steps)
            self.tensorboard.add_scalar(tag="Training/Bpp/Residues", scalar_value=encode_results["residues_bpp_avg"], global_step=self.train_steps)
            self.tensorboard.add_scalar(tag="Training/Bpp/Motion Info", scalar_value=encode_results["motion_bpp_avg"], global_step=self.train_steps)
# TODO: scalars

    def load_checkpoints(self):
        print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
        ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")
        start_epoch = 0
        best_rd_cost = 1e9
        stage = "inter_prediction"
        if self.args.checkpoints:
            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])
            try:
                stage = ckpt["stage"]
            except:
                print("Can not find stage indicator, just set to the first stage 'inter_prediction'.")
            try:
                self.optimizers[stage].load_state_dict(ckpt["optimizer"])
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
        # TODO: scheduler, aux_scheduler = self.init_scheduler(start_epoch=start_epoch)
        epoch_milestone = self.args.epoch_milestone
        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            if epoch < epoch_milestone[:1]:
                stage = "inter_prediction"
            elif epoch < sum(epoch_milestone[:2]):
                stage = "residues_compression"
            elif epoch < sum(epoch_milestone[:3]):
                stage = "total"
            else:
                stage = "rolling"

            print("\nEpoch {0}, stage is '{1}}'".format(str(epoch), stage))
            self.train_one_epoch(stage=stage)
            # TODO: scheduler.step()

            if stage is not "inter_prediction" and epoch % self.args.eval_epochs_interval == 0:
                rd_cost = self.evaluate()
                if epoch % self.args.save_epochs == 0 or rd_cost < best_rd_cost:
                    best_rd_cost = min(best_rd_cost, rd_cost)
                    self.save_ckpt(epoch=epoch, stage=stage, best_rd_cost=best_rd_cost)

    @torch.no_grad()
    def evaluate(self, stage) -> float:
        self.inter_frame_codec.eval()
        for frames in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode(frames, stage=stage)
            self.record.update('rd_cost', encode_results["rd_cost_avg_inter"].item())
            self.record.update('recon_psnr', encode_results["recon_psnr_avg"].item())
            self.record.update('recon_psnr_inter', encode_results["recon_psnr_avg_inter"].item())
            self.record.update('motion_bpp', encode_results["motion_bpp_avg"].item())
            self.record.update('residues_bpp', encode_results["residues_bpp_avg"].item())
            self.record.update('total_bpp', encode_results["total_bpp_avg"].item())
            self.record.update('total_bpp_inter', encode_results["total_bpp_avg_inter"].item())
        rd_cost = self.record.get('rd_cost', average=True)
        info = self.record.display()
        self.logger.info(info)
        return rd_cost

    def save_ckpt(self, stage: str, epoch: int, best_rd_cost: torch.Tensor):
        ckpt = {
            "inter_frame_codec": self.inter_frame_codec.state_dict(),
            "optimizer": self.optimizers[stage].state_dict(),
            "epoch": epoch,
            "stage": stage,
            "best_rd_cost": best_rd_cost
        }
        ckpt_path = "%s/codec_%.3d.pth" % (self.checkpoints_dir, epoch)
        torch.save(ckpt, ckpt_path)
        print("\nSave model to " + ckpt_path)


def get_aux_params(net: nn.Module):

    parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)

    # make sure there are no intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    # aux_optimizer = Adam(params=(params_dict[n] for n in sorted(list(aux_parameters))), lr=lr)
    params = (params_dict[n] for n in sorted(list(aux_parameters)))
    for n in sorted(list(aux_parameters)):
        print('aux_params = {}'.format(n))
    return params


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
