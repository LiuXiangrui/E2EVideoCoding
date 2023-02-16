from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from compressai.zoo import cheng2020_anchor as IntraFrameCodec
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from Common.Dataset import Vimeo90KDataset
from Common.Utils import Record, init
from Common.Utils import separate_aux_and_normal_params


class Trainer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, inter_frame_codec: nn.Module):
        self.args, self.logger, self.checkpoints_dir, self.tensorboard = init()
        self.record = Record(item_list=[
            'rd_cost',
            'recon_psnr', 'recon_psnr_inter',
            'motion_bpp', 'frame_bpp',
            'total_bpp', 'total_bpp_inter'
        ])

        self.inter_frame_codec = inter_frame_codec()
        self.intra_frame_codec = IntraFrameCodec(quality=self.args.intra_quality, metric="mse", pretrained=True)
        self.intra_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.inter_frame_codec.to("cuda" if self.args.gpu else "cpu")
        self.intra_frame_codec.eval()

        self.optimizers, self.aux_optimizers = self.init_optimizer()
        self.schedulers = self.aux_schedulers = None

        self.train_dataloader, self.eval_dataloader = self.init_dataloader()

        self.distortion_metric = nn.MSELoss()

        self.train_steps = 0

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, *args, **kwargs) -> float:
        self.inter_frame_codec.eval()
        for frames in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            encode_results = self.encode_sequence(frames, args, kwargs)
            for item in self.record.get_item_list():
                self.record.update(item, encode_results[item].item())
        rd_cost = self.record.get("rd_cost", average=True)
        info = self.record.display()
        self.logger.info(info)
        return rd_cost

    def train_one_epoch(self, *args, **kwargs):
        self.inter_frame_codec.train()
        for sequence in tqdm(self.train_dataloader, total=len(self.train_dataloader), smoothing=0.9, ncols=50):
            # encoding
            enc_results = self.encode_sequence(sequence, args, kwargs)
            # optimization
            self.optimize(enc_results, args, kwargs)
            # visualization
            self.train_steps += 1
            self.visualize(enc_results)
        self.lr_decay(args, kwargs)

    @abstractmethod
    def encode_sequence(self, frames: torch.Tensor, *args, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def optimize(self, enc_results: dict, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def visualize(self, enc_results: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def lr_decay(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def init_schedulers(self, start_epoch: int) -> tuple:
        raise NotImplementedError

    def init_dataloader(self):
        train_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_trainlist.txt", transform=RandomCrop(size=256))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
        eval_dataset = Vimeo90KDataset(root=self.args.dataset_root, list_filename="sep_testlist.txt")
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)
        return train_dataloader, eval_dataloader

    @abstractmethod
    def load_checkpoints(self):
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, epoch: int, best_rd_cost: torch.Tensor, *args, **kwargs) -> None:
        raise NotImplementedError


class TrainerOneStage(Trainer):
    def __init__(self, inter_frame_codec: nn.Module) -> None:
        super().__init__(inter_frame_codec=inter_frame_codec)

    def train(self) -> None:
        start_epoch, best_rd_cost = self.load_checkpoints()

        self.schedulers, self.aux_schedulers = self.init_schedulers(start_epoch=start_epoch)

        epoch_milestone = self.args.epoch_milestone
        assert len(epoch_milestone) == 1

        max_epochs = sum(epoch_milestone)
        for epoch in range(start_epoch, max_epochs):  # only ONE STAGE
            print("\nEpoch {0}".format(str(epoch)))
            self.train_one_epoch()

            if epoch % self.args.eval_epochs == 0:
                rd_cost = self.evaluate()
                if epoch % self.args.save_epochs == 0 or rd_cost < best_rd_cost:
                    best_rd_cost = min(best_rd_cost, rd_cost)
                    self.save_ckpt(epoch=epoch, best_rd_cost=best_rd_cost)

    @abstractmethod
    def encode_sequence(self, frames: torch.Tensor, *args, **kwargs) -> dict:
        raise NotImplementedError

    def optimize(self, enc_results: dict, *args, **kwargs) -> None:
        optimizer = self.optimizers[0]
        loss = enc_results["rd_cost"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        optimizer.step()
        optimizer.zero_grad()

        aux_optimizer = self.aux_optimizers[0]
        loss = enc_results["aux_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.inter_frame_codec.parameters(), max_norm=20)
        aux_optimizer.step()
        aux_optimizer.zero_grad()

    @abstractmethod
    def visualize(self, enc_results: dict) -> None:
        raise NotImplementedError

    def lr_decay(self, *args, **kwargs) -> None:
        self.schedulers[0].step()
        self.aux_schedulers[0].step()

    def init_optimizer(self) -> tuple:
        lr_milestone = self.args.lr_milestone
        assert len(lr_milestone) == 1
        params, aux_params = separate_aux_and_normal_params(self.inter_frame_codec)

        optimizer = Adam([{'params': params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])
        aux_optimizer = Adam([{'params': aux_params, 'initial_lr': lr_milestone[0]}], lr=lr_milestone[0])

        return [optimizer, ], [aux_optimizer, ]

    def init_schedulers(self, start_epoch: int) -> tuple:
        scheduler = MultiStepLR(optimizer=self.optimizers[0], milestones=self.args.lr_decay_milestone,
                                gamma=self.args.lr_decay_factor, last_epoch=start_epoch - 1)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizers[0], milestones=self.args.lr_decay_milestone,
                                    gamma=self.args.lr_decay_factor, last_epoch=start_epoch - 1)

        return [scheduler, ], [aux_scheduler, ]

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

    def save_ckpt(self, epoch: int, best_rd_cost: torch.Tensor, *args, **kwargs) -> None:
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
