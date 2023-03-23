import os
import matplotlib.pyplot as plt
import json


results_filepath = "../Results/All.json"

figure_save_folder = "../Figures"

color = ['#E91E63', 'red', '#ffb703', '#90be6d', '#219ebc', 'purple', '#3d5a80', 'grey', '#e29578', '#028f1e', '#cb0162', 'cornsilk']
marker = ['d', '*', 'o', 'v', 'X', 's', 'p', 'H', '^', 'P']

os.makedirs(figure_save_folder, exist_ok=True)

with open(results_filepath, mode='r') as f:
    data = json.load(f)
    for class_name in data.keys():
        plt.ylabel("PSNR (dB)")
        plt.xlabel("Bit per pixel")

        for i, method_name in enumerate(data[class_name].keys()):
            bpp_list = list(data[class_name][method_name]['bpp'])
            psnr_list = list(data[class_name][method_name]['psnr'])
            plt.plot(bpp_list, psnr_list,
                     color=color[min(i, len(color) - 1)], marker=marker[min(i, len(marker) - 1)],
                     markersize='6', lw=1.5, label=method_name
                     )

        plt.legend()

        plt.title(class_name)
        plt.savefig(os.path.join(figure_save_folder, class_name + ".png"), dpi=600)
        plt.close()

