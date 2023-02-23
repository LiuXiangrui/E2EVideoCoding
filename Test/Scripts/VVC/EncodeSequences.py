import json
import os
from multiprocessing import Pool

num_proc = 6

qp_list = [22, 27, 32, 37]

results_folder = r"C:\Users\XiangruiLiu\Desktop\VVC"

enc_cfg_path = r"C:\Users\XiangruiLiu\Desktop\encoder_lowdelay_P_vtm.cfg"
seq_cfg_folder = r"C:\Users\XiangruiLiu\Desktop\E2EVideoCoding\Test\Config\Sequences"

encoder_path = r"C:\Users\XiangruiLiu\Desktop\EncoderApp.exe"
decoder_path = r"C:\Users\XiangruiLiu\Desktop\DecoderApp.exe"


def construct_cmds(qp_: int):
    folder_per_qp = os.path.join(results_folder, "QP{}".format(str(qp_)))
    os.makedirs(folder_per_qp, exist_ok=True)

    enc_cmd_list = []
    dec_cmd_list = []

    for class_cfg in os.listdir(seq_cfg_folder):
        folder_per_class = os.path.join(folder_per_qp, os.path.splitext(class_cfg)[0])
        os.makedirs(folder_per_class, exist_ok=True)

        with open(os.path.join(seq_cfg_folder, class_cfg), 'r') as f:
            data = json.load(f)
            seq_folder = data["base_path"]
            seq_cfg_dict = data["sequences"]

            for seq_name in seq_cfg_dict.keys():
                rec_seq_path = os.path.join(folder_per_class, seq_name + ".yuv")
                results_path = os.path.join(folder_per_class, seq_name + ".txt")
                bin_path = os.path.join(folder_per_class, seq_name + ".bin")

                seq_path = os.path.join(seq_folder, seq_name + ".yuv")
                seq_cfg = seq_cfg_dict[seq_name]

                enc_args = {
                    "InputFile": seq_path,
                    "BitstreamFile": bin_path,
                    "ReconFile": rec_seq_path,
                    "SourceWidth": seq_cfg["SourceWidth"],
                    "SourceHeight": seq_cfg["SourceHeight"],
                    "OutputBitDepth": 8,
                    "FrameRate": seq_cfg["FrameRate"],
                    "FramesToBeEncoded": seq_cfg["FramesToBeEncoded"],
                    "Level": seq_cfg["Level"],
                    "QP": qp_,
                    "DecodingRefreshType": 2,
                    "IntraPeriod": 32,
                }

                enc_cmd = "{} -c {}".format(str(encoder_path), str(enc_cfg_path))

                for key, value in enc_args.items():
                    enc_cmd += " --{}={}".format(str(key), str(value))

                enc_cmd += " > {}".format(str(results_path))

                dec_cmd = "{} -b {} -o {} -d 8".format(str(decoder_path), str(bin_path), str(rec_seq_path))

                enc_cmd_list.append(enc_cmd)
                print(enc_cmd)
                dec_cmd_list.append(dec_cmd)
                input(dec_cmd)

    return enc_cmd_list, dec_cmd_list


def call_codec(cmd: str):
    # os.system(cmd)
    pass

def encode():
    enc_args_list = []
    dec_args_list = []
    for qp in qp_list:
        e, d = construct_cmds(qp)
        enc_args_list.extend(e)
        dec_args_list.extend(d)
    p = Pool(processes=num_proc)
    for args in enc_args_list:
        p.apply_async(call_codec, args=(args,))
    p.close()
    p.join()
    p = Pool(processes=num_proc)
    for args in dec_args_list:
        p.apply_async(call_codec, args=(args,))
    p.close()
    p.join()


if __name__ == "__main__":
    encode()

