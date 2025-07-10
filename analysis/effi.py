import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import font_manager as fm


def plot_all_graphs(dataset_names, bs_list, kd_list, dkd_list, x_list, y_label):
    font_prop = fm.FontProperties(family="Times New Roman", size=19)

    num_datasets = len(dataset_names)
    fig, axs = plt.subplots(
        num_datasets, 2, figsize=(12, 4 * num_datasets), sharey=False
    )
    letters = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
        "(i)",
        "(j)",
        "(k)",
        "(l)",
    ]

    for i, (dataset_name, bs, kd, dkd, x) in enumerate(
        zip(dataset_names, bs_list, kd_list, dkd_list, x_list)
    ):

        # Left column: SEQ vs OURS
        axs[i, 0].plot(bs, label="SEQ", marker="o", linestyle="-", color="darkblue")
        axs[i, 0].plot(dkd, label="OURS", marker="o", linestyle="-", color="darkgreen")
        # axs[i, 0].fill_between(x, np.array(dkd) - 0.01, np.array(dkd) + 0.01, alpha=0.2, color='darkgreen')
        axs[i, 0].set_ylabel(letters[i] + " " + dataset_name, fontproperties=font_prop)
        if i == 0:
            axs[i, 0].set_title(f"SEQ vs. OURS", fontproperties=font_prop)
            axs[i, 0].legend(loc="lower right", prop=font_prop)

        # Right column: KD vs OURS
        axs[i, 1].plot(kd, label="KD", marker="o", linestyle="-", color="darkorange")
        axs[i, 1].plot(dkd, label="OURS", marker="o", linestyle="-", color="darkgreen")
        # axs[i, 1].fill_between(x, np.array(dkd) - 0.01, np.array(dkd) + 0.01, alpha=0.2, color='darkgreen')
        if i==0:
            axs[i, 1].set_title(f"KD vs. OURS", fontproperties=font_prop)
            axs[i, 1].legend(loc="lower left", prop=font_prop)

        # Compute the min and max values for the y-axis based on the data points
        min_value_bs_dkd = min(min(bs), min(dkd))
        max_value_bs_dkd = max(max(bs), max(dkd))
        min_value_kd_dkd = min(min(kd), min(dkd))
        max_value_kd_dkd = max(max(kd), max(dkd))
        # Add labels next to each point with a vertical offset
        offset = 0.05  # Adjust as needed

        for j, (bs_val, kd_val, dkd_val) in enumerate(zip(bs, kd, dkd)):
            bs_dkd_val_max = bs_val if bs_val > dkd_val else dkd_val
            kd_dkd_val_max = kd_val if kd_val > dkd_val else dkd_val
            bs_dkd_val_min = bs_val if bs_val < dkd_val else dkd_val
            kd_dkd_val_min = kd_val if kd_val < dkd_val else dkd_val
            axs[i, 0].text(
                j,
                bs_dkd_val_min - offset * (max_value_bs_dkd - min_value_bs_dkd),
                f"{100*(bs_val if bs_val < dkd_val else dkd_val):.2f}",
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="center",
            )
            axs[i, 0].text(
                j,
                bs_dkd_val_max + offset * (max_value_bs_dkd - min_value_bs_dkd),
                f"{100*(bs_val if bs_val > dkd_val else dkd_val):.2f}",
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="center",
            )
            axs[i, 1].text(
                j,
                kd_dkd_val_min - offset * (max_value_kd_dkd - min_value_kd_dkd),
                f"{100*(kd_val if kd_val < dkd_val else dkd_val):.2f}",
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="center",
            )
            axs[i, 1].text(
                j,
                kd_dkd_val_max + offset * (max_value_kd_dkd - min_value_kd_dkd),
                f"{100*(kd_val if kd_val > dkd_val else dkd_val):.2f}",
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="center",
            )

        # Set x labels and ticks for all subplots
        axs[i, 0].set_xticks(np.arange(len(x)))
        axs[i, 0].set_xticklabels(x, rotation=0, fontproperties=font_prop)
        axs[i, 1].set_xticks(np.arange(len(x)))
        axs[i, 1].set_xticklabels(x, rotation=0, fontproperties=font_prop)

        # Add a margin for better visualization
        margin_bs_dkd = (max_value_bs_dkd - min_value_bs_dkd) * 0.1
        margin_kd_dkd = (max_value_kd_dkd - min_value_kd_dkd) * 0.1

        # Set the y-axis limits
        axs[i, 0].set_ylim(
            min_value_bs_dkd - margin_bs_dkd, max_value_bs_dkd + margin_bs_dkd
        )
        axs[i, 1].set_ylim(
            min_value_kd_dkd - margin_kd_dkd, max_value_kd_dkd + margin_kd_dkd
        )

    plt.tight_layout()
    plt.savefig("all_datasets_dice.pdf")
    plt.show()


"""
The test results are copied from the analysis/table_figure.py

"""

# Usage of the function with multiple datasets
plot_all_graphs(
    dataset_names=[
        "Prostate",
        "Optic-cup",
        "Optic-disc",
        "LV-endo",
        "LV-epi",
        "RV",
        "Hippocampus",
    ],
    bs_list=[
        [0.86179, 0.38169, 0.76494, 0.84482, 0.8046, 0.698],
        [0.80266, 0.75621, 0.61493, 0.58947],
        [0.80884, 0.73047, 0.69424, 0.59210],
        [0.87913, 0.87149, 0.89304, 0.86135],
        [0.79318, 0.79189, 0.80048, 0.7459],
        [0.849, 0.81111, 0.82853, 0.772],
        [0.85797, 0.78323, 0.77932],
    ],
    kd_list=[
        [0.86179, 0.74187, 0.67121, 0.8312, 0.83898, 0.68242],
        [0.80266, 0.78931, 0.64175, 0.49151],
        [0.80884, 0.68654, 0.69103, 0.62330],
        [0.87913, 0.88541, 0.89615, 0.87682],
        [0.79317, 0.79949, 0.80872, 0.78015],
        [0.84900, 0.82691, 0.82805, 0.79637],
        [0.85797, 0.80696, 0.79512],
    ],
    dkd_list=[
        [0.86903, 0.8490, 0.8164, 0.8592, 0.8479, 0.8140],
        [0.81857, 0.80870, 0.76042, 0.64545],
        [0.80231, 0.74022, 0.76685, 0.69323],
        [0.87273, 0.87679, 0.88128, 0.88746],
        [0.80036, 0.80823, 0.81083, 0.79394],
        [0.84282, 0.82072, 0.82329, 0.77767],
        [0.86288, 0.83660, 0.81662],
    ],
    x_list=[
        ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"],
        ["Zeiss", "RIM", "Drishti", "Canon"],
        ["Zeiss", "RIM", "Drishti", "Canon"],
        ["Siemens", "Philips", "GE", "Canon"],
        ["Siemens", "Philips", "GE", "Canon"],
        ["Siemens", "Philips", "GE", "Canon"],
        ["Decathlon", "Dryad", "Harp"],
    ],
    y_label=" DICE of ",
)
