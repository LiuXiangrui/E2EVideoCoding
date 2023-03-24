from abc import ABCMeta, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
from compressai.zoo import cheng2020_anchor as IntraFrameCodec
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from tqdm import tqdm

from Model.Common.Dataset import Vimeo90KDataset, Vimeo90KDatasetDVC
from Model.Common.Utils import Record, init, clip_gradients

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class TrainerABC(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.network_args, self.training_args, self.logger, self.checkpoints_dir, self.tensorboard = init()
        self.record = Record(item_list=self.training_args.record_items) if not self.training_args.disable_eval else None

        self.inter_frame_codec = None
        self.intra_frame_codec = IntraFrameCodec(quality=self.training_args.intra_quality, metric="mse", pretrained=True)

        self.optimizers = self.aux_optimizers = None
        self.schedulers = self.aux_schedulers = None

        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

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

        self.optimizers, self.aux_optimizers = self.init_optimizer()

        start_epoch, best_rd_cost = self.load_checkpoints()

        self.best_rd_cost_per_stage[self.infer_stage(epoch=start_epoch)] = best_rd_cost

        self.schedulers, self.aux_schedulers = self.init_schedulers(start_epoch=start_epoch)

        epoch_milestone = self.training_args.epoch_milestone

        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            stage = self.infer_stage(epoch)
            print("\nEpoch {0} Stage '{1}".format(str(epoch), stage))
            self.train_one_epoch(stage=stage)

            if self.training_args.disable_eval:  # skip evaluation
                if epoch % self.training_args.save_epochs == 0:
                    self.save_checkpoints(epoch=epoch, best_rd_cost=self.best_rd_cost_per_stage[stage], stage=stage)
            else:
                if epoch % self.training_args.eval_epochs == 0:
                    rd_cost = self.evaluate(stage=stage)
                    if rd_cost < self.best_rd_cost_per_stage[stage]:
                        self.best_rd_cost_per_stage[stage] = min(self.best_rd_cost_per_stage[stage], rd_cost)
                        self.save_checkpoints(epoch=epoch, best_rd_cost=self.best_rd_cost_per_stage[stage], stage=stage)

    @torch.no_grad()
    def evaluate(self, stage: Enum) -> float:
        self.inter_frame_codec.eval()
        for frames in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode_sequence(frames, stage=stage)
            for item in self.record.get_item_list():
                self.record.update(item, encode_results[item].item())
        rd_cost = self.record.get("rd_cost", average=True)
        info = self.record.display()
        self.tensorboard.add_text(tag="Test", text_string=info, global_step=self.eval_epoch)
        self.eval_epoch += 1
        self.logger.info(info)
        return rd_cost

    def train_one_epoch(self, stage: Enum) -> None:
        self.inter_frame_codec.train()
        for sequence in tqdm(self.train_dataloader, total=len(self.train_dataloader), smoothing=0.9, ncols=50):
            enc_results = self.encode_sequence(sequence, stage=stage)

            self.optimize(enc_results, stage=stage)

            self.train_steps += 1
            self.visualize(enc_results, stage=stage)

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

    def init_dataloader(self) -> tuple:
        if self.training_args.disable_eval:
            train_dataloader = DataLoader(dataset=Vimeo90KDatasetDVC(root=self.training_args.dataset_root,
                                                                     split_filepath=self.training_args.split_filepath,
                                                                     transform=Compose([
                                                                       RandomCrop(size=256), RandomHorizontalFlip(p=0.5),
                                                                       RandomVerticalFlip(p=0.5)
                                                                     ])),
                                          batch_size=self.training_args.batch_size, shuffle=True)
            eval_dataloader = None
        else:
            train_dataloader = DataLoader(dataset=Vimeo90KDataset(root=self.training_args.dataset_root,
                                                                  list_filename="sep_trainlist.txt",
                                                                  transform=Compose([
                                                                    RandomCrop(size=256),
                                                                    RandomHorizontalFlip(p=0.5),
                                                                    RandomVerticalFlip(p=0.5)
                                                                  ])),
                                          batch_size=self.training_args.batch_size, shuffle=True)
            eval_dataloader = DataLoader(dataset=Vimeo90KDataset(root=self.training_args.dataset_root,
                                                                 list_filename="sep_testlist.txt"),
                                         batch_size=1, shuffle=False)
        return train_dataloader, eval_dataloader

    def load_checkpoints(self) -> tuple:
        start_epoch = 0
        best_rd_cost = 1e9
        if hasattr(self.training_args, "checkpoints"):
            print("\n===========Load checkpoints {0}===========\n".format(self.training_args.checkpoints))
            ckpt = torch.load(self.training_args.checkpoints, map_location="cuda" if self.training_args.gpu else "cpu")

            best_rd_cost = ckpt['best_rd_cost']

            start_epoch = ckpt["epoch"] + 1

            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])

            stage = self.infer_stage(epoch=ckpt["epoch"])
            self.optimizers[stage].load_state_dict(ckpt["optimizer"])
            self.aux_optimizers[stage].load_state_dict(ckpt["aux_optimizer"])
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
        return start_epoch, best_rd_cost

    def save_checkpoints(self, epoch: int, best_rd_cost: torch.Tensor, stage: Enum) -> None:
        ckpt = {
            "inter_frame_codec": self.inter_frame_codec.state_dict(),
            "optimizer": self.optimizers[stage].state_dict(),
            "aux_optimizer": self.aux_optimizers[stage].state_dict(),
            "epoch": epoch,
            "best_rd_cost": best_rd_cost
        }
        ckpt_path = "%s/Inter_Frame_Codec_%.3d.pth" % (self.checkpoints_dir, epoch)
        torch.save(ckpt, ckpt_path)
        print("\nSave model to " + ckpt_path)
