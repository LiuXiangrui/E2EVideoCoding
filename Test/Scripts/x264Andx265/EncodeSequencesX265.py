import json
import os
from multiprocessing import Pool

qp_list = [22, 27, 32, 37]

results_folder = r"D:\Traditional\x265"

seq_cfg_folder = r"..\Config\Sequences"

encoder_path = r"C:\Users\xiangrliu3\Desktop\ffmpeg.exe"
decoder_path = r"C:\Users\xiangrliu3\Desktop\ffmpeg.exe"


def construct_cmds(qp_: int):
    folder_per_qp = os.path.join(results_folder, "QP{}".format(str(qp_)))
    os.makedirs(folder_per_qp, exist_ok=True)

    enc_cmd_list = []

    for class_cfg in os.listdir(seq_cfg_folder):
        folder_per_class = os.path.join(folder_per_qp, os.path.splitext(class_cfg)[0])
        os.makedirs(folder_per_class, exist_ok=True)

        with open(os.path.join(seq_cfg_folder, class_cfg), 'r') as f:
            data = json.load(f)
            seq_folder = data["base_path"]
            seq_cfg_dict = data["sequences"]

            for seq_name in seq_cfg_dict.keys():
                results_path = os.path.join(folder_per_class, seq_name + ".txt")
                bin_path = os.path.join(folder_per_class, seq_name + ".mkv")

                seq_path = os.path.join(seq_folder, seq_name + ".yuv")
                seq_cfg = seq_cfg_dict[seq_name]

                enc_cmd = r"{} -pix_fmt yuv420p -s {}x{} " \
                          "-r {} -i {} -vframes {} -c:v libx265 -preset veryslow -tune zerolatency -tune psnr " \
                          "-x265-params \"qp={}:keyint={}\" " \
                          "-flags +psnr -y " \
                          "{} > {} 2>&1".format(encoder_path, seq_cfg["SourceWidth"], seq_cfg["SourceHeight"],
                                                seq_cfg["FrameRate"], seq_path, seq_cfg["FramesToBeEncoded"],
                                                qp_, seq_cfg["GOPSize"],
                                                bin_path, results_path)
                enc_cmd_list.append(enc_cmd)

    return enc_cmd_list


def call_codec(args: str):
    os.system(args)


def encode():
    enc_args_list = []
    for qp in qp_list:
        e = construct_cmds(qp)
        enc_args_list.extend(e)
    p = Pool(processes=8)
    for args in enc_args_list:
        p.apply_async(call_codec, args=(args,))
    p.close()
    p.join()


if __name__ == "__main__":
    encode()
