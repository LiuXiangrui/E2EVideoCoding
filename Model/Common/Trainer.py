from abc import ABCMeta, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
from compressai.zoo import cheng2020_anchor as IntraFrameCodec
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from Model.Common.Dataset import Vimeo90KDataset
from Model.Common.Utils import Record, init

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class TrainerABC(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.args, self.logger, self.checkpoints_dir, self.tensorboard = init()
        self.record = Record(item_list=self.args.record_items)

        self.inter_frame_codec = None
        self.intra_frame_codec = IntraFrameCodec(quality=self.args.intra_quality, metric="mse", pretrained=True)

        self.optimizers = self.aux_optimizers = None
        self.schedulers = self.aux_schedulers = None

        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

        self.distortion_metric = nn.MSELoss()

        self.train_steps = 0

    def train(self) -> None:
        self.intra_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.inter_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.intra_frame_codec.eval()

        self.optimizers, self.aux_optimizers = self.init_optimizer()

        start_epoch, best_rd_cost = self.load_checkpoints()

        self.schedulers, self.aux_schedulers = self.init_schedulers(start_epoch=start_epoch)

        epoch_milestone = self.args.epoch_milestone if isinstance(self.args.epoch_milestone, list) else [self.args.epoch_milestone, ]

        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):
            stage = self.infer_stage(epoch)
            print("\nEpoch {0} Stage '{1}".format(str(epoch), stage))
            self.train_one_epoch(stage=stage)

            if epoch % self.args.eval_epochs == 0:
                rd_cost = self.evaluate(stage=stage)
                if epoch % self.args.save_epochs == 0 or rd_cost < best_rd_cost:
                    best_rd_cost = min(best_rd_cost, rd_cost)
                    self.save_checkpoints(epoch=epoch, best_rd_cost=best_rd_cost, stage=stage)

    @torch.no_grad()
    def evaluate(self, stage: Enum) -> float:
        self.inter_frame_codec.eval()
        for frames in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode_sequence(frames, stage=stage)
            for item in self.record.get_item_list():
                self.record.update(item, encode_results[item].item())
        rd_cost = self.record.get("rd_cost", average=True)
        info = self.record.display()
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
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        optimizer.step()
        optimizer.zero_grad()

        aux_optimizer = self.aux_optimizers[stage]
        loss = enc_results["aux_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        aux_optimizer.step()
        aux_optimizer.zero_grad()

    @abstractmethod
    def visualize(self, enc_results: dict, stage: Enum) -> None:
        raise NotImplementedError

    def lr_decay(self, stage: Enum) -> None:
        self.schedulers[stage].step()
        self.aux_schedulers[stage].step()

    @abstractmethod
    def infer_stage(self, epoch: int) -> Enum:
        raise NotImplementedError

    @abstractmethod
    def init_optimizer(self) -> tuple[dict, dict]:
        raise NotImplementedError

    @abstractmethod
    def init_schedulers(self, start_epoch: int) -> tuple:
        raise NotImplementedError

    def init_dataloader(self) -> tuple:
        train_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_trainlist.txt",
                                        transform=RandomCrop(size=256))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
        eval_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_testlist.txt")
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)
        return train_dataloader, eval_dataloader

    def load_checkpoints(self) -> tuple:
        start_epoch = 0
        best_rd_cost = 1e9
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")

            best_rd_cost = ckpt['best_rd_cost']

            start_epoch = ckpt["epoch"] + 1

            self.inter_frame_codec.load_state_dict(ckpt["inter_frame_codec"])

            stage = self.infer_stage(epoch=ckpt["epoch"])
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
