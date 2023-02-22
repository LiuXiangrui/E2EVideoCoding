import os
import matplotlib.pyplot as plt
import json


results_folder = "./Results"

figure_save_folder = "./Figures"

class_name_list = ["HEVC_CLASS_B", "HEVC_CLASS_C", "HEVC_CLASS_D", "HEVC_CLASS_E", "MCL-JCV", "UVG"]

os.makedirs(figure_save_folder, exist_ok=True)

for class_name in class_name_list:
    for method in os.listdir(results_folder):
        with open(os.path.join(results_folder, method, class_name + ".json"), mode='r') as f:
            data = json.load(f)[class_name]
            plt.plot(data["bpp"], data["psnr"], linewidth=2, markersize=6, label=method)
    plt.legend()
    plt.ylabel("PSNR (dB)")
    plt.xlabel("Bit per pixel")
    plt.title(class_name)
    plt.savefig(os.path.join(figure_save_folder, class_name + ".png"), dpi=600)
    plt.close()

