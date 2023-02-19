import json
import os
from multiprocessing import Pool

qp_list = [22, 27, 32, 37]

results_folder = r"D:\Traditional"

enc_cfg_prefix = r"encoder_lowdelay_P_main"
seq_cfg_folder = r"C:\Users\xiangrliu3\Desktop\E2EVideoCoding\Test\Config\seq2"

encoder_path = r"TAppEncoder.exe"
decoder_path = r"TAppDecoder.exe"


def construct_cmds(qp_: int):
    folder_per_qp = os.path.join(results_folder, "QP{}".format(str(qp_)))
    os.makedirs(folder_per_qp, exist_ok=True)

    enc_args_list = []
    dec_args_list = []

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

                gop_size = str(seq_cfg["GOPSize"])
                enc_cfg_path = enc_cfg_prefix + "_GOP_{}.cfg".format(gop_size)

                if "LastValidFrame" in seq_cfg:

                    enc_args = "{} -c {} -i {} -b {} -o {} " \
                           "-f {} -fr {} " \
                           "-wdt {} -hgt {} " \
                           "-q {} -g {} --LastValidFrame={} --Level={} --OutputBitDepth=8 > {}".format(
                        str(encoder_path), str(enc_cfg_path), str(seq_path), str(bin_path), str(rec_seq_path),
                        str(seq_cfg["FramesToBeEncoded"]), str(seq_cfg["FrameRate"]),
                        str(seq_cfg["SourceWidth"]), str(seq_cfg["SourceHeight"]),
                        str(qp_), str(seq_cfg["GOPSize"]), str(seq_cfg["LastValidFrame"]),str(seq_cfg["Level"]), str(results_path))
                else:
                    enc_args = "{} -c {} -i {} -b {} -o {} " \
                               "-f {} -fr {} " \
                               "-wdt {} -hgt {} " \
                               "-q {} -g {} --Level={} --OutputBitDepth=8 > {}".format(
                        str(encoder_path), str(enc_cfg_path), str(seq_path), str(bin_path), str(rec_seq_path),
                        str(seq_cfg["FramesToBeEncoded"]), str(seq_cfg["FrameRate"]),
                        str(seq_cfg["SourceWidth"]), str(seq_cfg["SourceHeight"]),
                        str(qp_), str(seq_cfg["GOPSize"]), str(seq_cfg["Level"]),
                        str(results_path))

                dec_args = "{} -b {} -o {} -d 8".format(str(decoder_path), str(bin_path), str(rec_seq_path))

                enc_args_list.append(enc_args)
                dec_args_list.append(dec_args)

    return enc_args_list, dec_args_list


def call_codec(args: str):
    os.system(args)


def encode():
    enc_args_list = []
    dec_args_list = []
    for qp in qp_list:
        e, d = construct_cmds(qp)
        enc_args_list.extend(e)
        dec_args_list.extend(d)
    p = Pool(processes=8)
    for args in enc_args_list:
        p.apply_async(call_codec, args=(args,))
    p.close()
    p.join()
    p = Pool(processes=8)
    for args in dec_args_list:
        p.apply_async(call_codec, args=(args,))
    p.close()
    p.join()


if __name__ == "__main__":
    encode()
