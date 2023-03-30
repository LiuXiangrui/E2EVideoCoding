import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Vimeo90KDataset(Dataset):  # modified based on PytorchCompression
    def __init__(self, root: str, training_frames_list_path: str, use_all_frames: bool = False, transform=None) -> None:

        self.use_all_frames = use_all_frames  # use all reference in the rolling stage in training

        self.to_tensor = ToTensor()
        self.transform = transform

        self.frames_list = []
        with open(training_frames_list_path) as f:
            data = f.readlines()
        for line in data:
            frame_filepath = os.path.join(root, line.rstrip())
            if use_all_frames:
                sequence_dir = os.path.split(frame_filepath)[0]
                self.frames_list.append([os.path.join(sequence_dir, frame_name) for frame_name in os.listdir(sequence_dir)])
            else:
                reference_distance = 2
                frame_filepath = os.path.join(root, line.rstrip())
                reference_idx = int(os.path.splitext(frame_filepath)[0][-1]) - reference_distance
                reference_filepath = "{}.png".format(frame_filepath[:-5] + str(reference_idx))
                self.frames_list.append([reference_filepath, frame_filepath])

    def __len__(self) -> int:
        return len(self.frames_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        frames = torch.stack([self.to_tensor(Image.open(frame_path).convert("RGB")) for frame_path in self.frames_list[index]], dim=0)

        if self.transform:
            return self.transform(frames)
        return frames
