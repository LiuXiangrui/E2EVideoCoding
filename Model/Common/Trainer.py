from abc import ABCMeta, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
from compressai.zoo import cheng2020_anchor as IntraFrameCodec
from tqdm import tqdm

from Model.Common.Utils import Record, init, clip_gradients

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class TrainerABC(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.network_args, self.training_args, self.logger, self.checkpoints_dir, self.tensorboard = init()
        self.record = None if self.training_args.disable_eval else Record(item_list=self.training_args.record_items)

        self.inter_frame_codec = None
        self.intra_frame_codec = IntraFrameCodec(quality=self.training_args.intra_quality, metric="mse", pretrained=True)

        self.optimizers = self.aux_optimizers = None
        self.schedulers = self.aux_schedulers = None

        self.train_dataloaders, self.eval_dataloader = self.init_dataloader()

        if self.training_args.metric == "mse":
            self.distortion_metric = nn.MSELoss()
        else:
            self.distortion_metric = None

        self.train_steps = self.eval_epoch = 0

        self.best_rd_cost_per_stage = None

    def train(self) -> None:
        self.intra_frame_codec.to("cuda" if self.training_args.gpu else "cpu")
        self.inter_frame_codec.to("cuda" if self.training_args.gpu else "cpu")
        self.intra_frame_codec.eval()
        # initialize optimizers
        self.optimizers, self.aux_optimizers = self.init_optimizer()

        start_epoch = self.load_checkpoints()
        # initialize learning rate decay schedulers
        self.schedulers, self.aux_schedulers = self.init_schedulers(start_epoch=start_epoch)

        epoch_milestone = self.training_args.epoch_milestone

        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            # specify current stage based on the current epoch
            stage = self.infer_stage(epoch)
            print("\nEpoch {0} Stage '{1}".format(str(epoch), stage))
            # training one epoch
            self.train_one_epoch(stage=stage)
            # save checkpoints
            if epoch % self.training_args.save_epochs == 0:
                self.save_checkpoints(epoch=epoch, stage=stage)

    def train_one_epoch(self, stage: Enum) -> None:
        self.inter_frame_codec.train()
        train_dataloader = self.train_dataloaders[stage]
        for sequence in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9, ncols=50):
            # forward propagation
            enc_results = self.encode_sequence(sequence, stage=stage)
            # backward propagation and optimize
            self.optimize(enc_results, stage=stage)
            self.train_steps += 1
            # visualize by tensorboard
            if self.training_args.verbose:
                self.visualize(enc_results, stage=stage)
        # learning rate decay
        self.lr_decay(stage=stage)

    @abstractmethod
    def encode_sequence(self, frames: torch.Tensor, stage: Enum) -> dict:
        raise NotImplementedError

    def optimize(self, enc_results: dict, stage: Enum) -> None:
        optimizer = self.optimizers[stage]
        loss = enc_results["rd_cost"]
        loss.backward()
        clip_gradients(optimizer, grad_clip=self.training_args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        aux_optimizer = self.aux_optimizers[stage]
        loss = enc_results["aux_loss"]
        loss.backward()
        aux_optimizer.step()
        aux_optimizer.zero_grad()

    @abstractmethod
    def visualize(self, enc_results: dict, stage: Enum) -> None:
        raise NotImplementedError

    @abstractmethod
    def lr_decay(self, stage: Enum) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer_stage(self, epoch: int) -> Enum:
        raise NotImplementedError

    @abstractmethod
    def init_optimizer(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def init_schedulers(self, start_epoch: int) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def init_dataloader(self) -> tuple:
        raise NotImplementedError

    def load_checkpoints(self) -> int:
        start_epoch = 0
        # load checkpoints and resume training
        if hasattr(self.training_args, "checkpoints"):
            print("\n===========Load checkpoints {0}===========\n".format(self.training_args.checkpoints))
            ckpt = torch.load(self.training_args.checkpoints, map_location="cuda" if self.training_args.gpu else "cpu")
            # load record
            start_epoch = ckpt["epoch"] + 1
            stage = self.infer_stage(epoch=ckpt["epoch"])
            # load parameters
            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])
            self.optimizers[stage].load_state_dict(ckpt["optimizer"])
            self.aux_optimizers[stage].load_state_dict(ckpt["aux_optimizer"])
        # load pretrained weights and training from start
        elif hasattr(self.training_args, "pretrained"):
            ckpt = torch.load(self.training_args.pretrained)
            print("\n===========Load pretrained {0}===========\n".format(self.training_args.pretrained))
            pretrained_dict = ckpt["inter_frame_codec"]
            model_dict = self.inter_frame_codec.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.inter_frame_codec.load_state_dict(model_dict)
        else:
            print("\n===========Training from scratch===========\n")
        return start_epoch

    def save_checkpoints(self, epoch: int, stage: Enum) -> None:
        ckpt = {
            "inter_frame_codec": self.inter_frame_codec.state_dict(),
            "optimizer": self.optimizers[stage].state_dict(),
            "aux_optimizer": self.aux_optimizers[stage].state_dict(),
            "epoch": epoch,
        }
        ckpt_path = "%s/Inter_Frame_Codec_%.3d.pth" % (self.checkpoints_dir, epoch)
        torch.save(ckpt, ckpt_path)
        print("\nSave model to " + ckpt_path)
