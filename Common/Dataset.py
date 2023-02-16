import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Vimeo90KDataset(Dataset):
    def __init__(self, root: str, list_filename: str, seq_folder: str = "sequences", transform=None):
        self.to_tensor = ToTensor()
        self.transform = transform
        self.num_available_frames = 7

        with open(os.path.join(root, list_filename), mode='r') as f:
            seq_list = f.readlines()
            seq_list = [os.path.join(root, seq_folder, seq.strip('\n')) for seq in seq_list]

        self.seq_list = seq_list

    def __getitem__(self, index) -> torch.Tensor:
        frames_path = [os.path.join(self.seq_list[index], "im{0}.png".format(str(i))) for i in range(1, self.num_available_frames + 1)]

        frames = torch.stack([self.to_tensor(Image.open(frame).convert("RGB"))[:, :512, :512] for frame in frames_path], dim=0)

        if self.transform:
            return self.transform(frames)
        return frames

    def __len__(self) -> int:
        return len(self.seq_list)
