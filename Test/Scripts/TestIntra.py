from compressai.zoo import cheng2020_anchor as IntraFrameCodec

from Model.Common.Utils import write_uintx, read_uintx, write_bytes, read_bytes

intra_frame_codec = IntraFrameCodec(quality=5, metric="mse", pretrained=True)

intra_frame_codec.eval()

from PIL import Image
import numpy as np
import torch


frame = Image.open(r"D:\ImageDataset\DIV2K_train_HR\0001.png")

frame = torch.from_numpy(np.array(frame))

# scale to [0, 1]
frame = frame / 255.

frame = frame.unsqueeze(dim=0).permute(0, 3, 1, 2)


height, width = frame.shape[2:]

# padding





with torch.no_grad():
    enc_results = intra_frame_codec.compress(frame)

    shape = enc_results["shape"]
    strings = enc_results["strings"]

    dec_results = intra_frame_codec.decompress(strings=strings, shape=shape)

    dec_img = dec_results["x_hat"]
    dec_img = dec_img.permute(0, 2, 3, 1).squeeze(dim=0).numpy() * 255.
    dec_img = Image.fromarray(dec_img.astype(np.uint8))
    dec_img.show()

    exit()


    with open("1.bin", mode='wb') as f:
        for value in shape:  # write the shape of features needed to be decoded by entropy bottleneck
            write_uintx(f, value=value, bit_depth=16)

        num_string = len(strings)
        write_uintx(f, value=num_string, bit_depth=8)  # write how many strings need to write

        for string in strings:
            string = string[0]  # note that string is a list containing 1 element, and I don't know why?
            len_string = len(string)
            write_uintx(f, value=len_string, bit_depth=32)  # write the length of the string
            write_bytes(f, values=string)  # write the string

    with open("1.bin", mode="rb") as f:
        shape = [read_uintx(f, bit_depth=16) for _ in range(2)]

        num_string = read_uintx(f, bit_depth=8)

        strings = [[read_bytes(f, read_uintx(f, bit_depth=32)), ] for _ in range(num_string)]

    dec_results = intra_frame_codec.decompress(strings=strings, shape=shape)

    dec_img = dec_results["x_hat"]
    dec_img = dec_img.permute(0, 2, 3, 1).squeeze(dim=0).numpy() * 255.
    dec_img = Image.fromarray(dec_img.astype(np.uint8))
    dec_img.show()

    print(1)



