import json
import os

enc_results_folder = r"D:\HEVCResults"
save_folder = r"D:\HEVCResults"

qp_list = [22, 27, 32, 37]
class_name_list = ["HEVC_CLASS_B", "HEVC_CLASS_C", "HEVC_CLASS_D", "HEVC_CLASS_E", "MCL-JCV", "UVG"]


def parse_encoder_results(results_path: str) -> dict:
    with open(results_path, mode='r') as f:
        raw_data = f.readlines()
        psnr_per_frame = []
        bits_per_frame = []
        resolution = frame_rate = 0

        for line in raw_data:
            if "Real     Format" in line:
                items = line.split()
                resolution = int(items[-2].split('x')[0]) * int(items[-2].split('x')[1])
                frame_rate = int(items[-1][:-2])

            if "POC" in line:
                raw_data_per_frame = line.split()
                for idx, item in enumerate(raw_data_per_frame):
                    if item == "bits":
                        bits_per_frame.append(float(raw_data_per_frame[idx - 1]))
                    if item == "[Y":
                        psnr_per_frame.append([  # [Y-PSNR, U-PSNR, V-PSNR]
                            float(raw_data_per_frame[idx + 1]),
                            float(raw_data_per_frame[idx + 4]),
                            float(raw_data_per_frame[idx + 7])
                        ])
        raw_data_seq = raw_data[-21].split()
        bitrate_kbps = float(raw_data_seq[2])
        bpp = bitrate_kbps * 1000 / frame_rate / resolution  # TODO: is it right?

        psnr = [float(raw_data_seq[3]), float(raw_data_seq[4]), float(raw_data_seq[5]), float(raw_data_seq[6])]  # [Y-PSNR, U-PSNR, V-PSNR, YUV-PSNR]

        return {
            "sequence_name": os.path.split(results_path)[-1].split('_')[0],
            "bpp": bpp,
            "psnr": psnr,
            "bits_per_frame": bits_per_frame,
            "psnr_per_frame": psnr_per_frame,
        }


def parse_all_classes():
    for class_name in class_name_list:
        sequences_results = {}
        for qp in qp_list:
            class_folder = os.path.join(enc_results_folder, "QP{}".format(str(qp)), class_name)
            for enc_results in os.listdir(class_folder):
                if os.path.splitext(enc_results)[-1] != ".txt":
                    continue
                parsed_results = parse_encoder_results(results_path=os.path.join(class_folder, enc_results))
                sequence_name = parsed_results["sequence_name"]
                bitrate = parsed_results["bpp"]
                psnr = parsed_results["psnr"][-1]  # note: only YUV-PSNR are recorded here

                if sequence_name in sequences_results:
                    sequences_results[sequence_name].append([bitrate, psnr])
                else:
                    sequences_results[sequence_name] = [[bitrate, psnr], ]

        avg_psnr_list = [0] * len(qp_list)
        avg_bpp_list = [0] * len(qp_list)

        rd_points = {}

        for sequence_name in sequences_results:
            bpp_list = []
            psnr_list = []
            for qp_idx, item in enumerate(sequences_results[sequence_name]):
                bpp_list.append(item[0])
                psnr_list.append(item[1])
                avg_bpp_list[qp_idx] += item[0] / len(sequences_results)
                avg_psnr_list[qp_idx] += item[1] / len(sequences_results)
                rd_points[sequence_name] = {"bpp": bpp_list, "psnr": psnr_list}

        # TODO: just average all sequences?
        rd_points[class_name] = {"bpp": avg_bpp_list, "psnr": avg_psnr_list}

        with open(os.path.join(save_folder, class_name + ".json"), mode='w') as f:
            json.dump(rd_points, f)


if __name__ == "__main__":
    os.makedirs(save_folder, exist_ok=True)
    parse_all_classes()
