import os
import torch


def slim_models(model_folder: str, dst_folder: str):
    os.makedirs(dst_folder, exist_ok=True)
    for ckpt_name in os.listdir(model_folder):
        if os.path.splitext(ckpt_name)[-1] != ".pth":
            continue
        ckpt = torch.load(os.path.join(model_folder, ckpt_name))
        ckpt = {"inter_frame_codec": ckpt["inter_frame_codec"]}
        torch.save(ckpt, os.path.join(dst_folder, ckpt_name))


if __name__ == "__main__":
    slim_models(r'C:\Users\xiangrliu3\Desktop\WithRolling', r'C:\Users\xiangrliu3\Desktop\WithRolling')
