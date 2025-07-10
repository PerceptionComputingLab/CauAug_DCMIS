import os
import sys

from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
from mp.eval.evaluate import ds_metrics
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

if __name__ == "__main__":

    features_seq_old = []
    features_seq_new = []
    distance_seq = []
    features_kd_old = []
    features_kd_new = []
    distance_kd = []
    features_cdd_old = []
    features_cdd_new = []
    distance_cdd = []

    dataset = "mm"
    approach = "seq"
    backbone = "unet"
    config["experiment_name"] = dataset + "-i" + "-" + approach + "-" + backbone
    config["approach"] = approach
    config["dataset"] = dataset
    config["resume_epoch"] = 40
    config["device"] = "cuda:5"
    # for baseline
    exp = Experiment(
        config=config,
        name=config["experiment_name"],
        notes="",
        reload_exp=(config["resume_epoch"]),  # TODO
    )
    dataset_list = []
    train_dataloader, _, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
    for ds_name, _ in datasets.items():
        dataset_list.append(ds_name[0])
    train_dataset = train_dataloader[1]
    train_dataset_old = train_dataloader[0]

    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")
    best_states = []
    with open(best_states_file, "r") as f:
        for line in f.readlines():
            best_states.append(int(line.replace("\n", "")))

    model = get_model(config, nr_labels=label_inf["label_nr"])

    agent_seq = get_agent(config, model=model, label_names=label_inf["label_names"])

    state_old = best_states[0]
    agent_seq.restore_state(exp_run.paths["states"], state_old)
    agent_seq.model.finish()
    state_new = best_states[-1]
    agent_seq.restore_state(exp_run.paths["states"], state_new)

    # for kd
    config["experiment_name"] = dataset + "-i" + "-" + "kd" + "-" + backbone
    config["approach"] = "cauaug"
    exp = Experiment(
        config=config,
        name=config["experiment_name"],
        notes="",
        reload_exp=(config["resume_epoch"]),  # TODO
    )
    _, _, _, exp_run, _ = get_dataset(config, exp=exp)
    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")
    best_states = []
    with open(best_states_file, "r") as f:
        for line in f.readlines():
            best_states.append(int(line.replace("\n", "")))

    model = get_model(config, nr_labels=label_inf["label_nr"])

    agent_kd = get_agent(config, model=model, label_names=label_inf["label_names"])

    state_old = best_states[0]
    agent_kd.restore_state(exp_run.paths["states"], state_old)
    agent_kd.model.finish()
    state_new = best_states[-1]
    agent_kd.restore_state(exp_run.paths["states"], state_new)

    # for 
    config["experiment_name"] = dataset + "-i" + "-" + "dom" + "-" + backbone
    config["approach"] = "cauaug"
    exp = Experiment(
        config=config,
        name=config["experiment_name"],
        notes="",
        reload_exp=(config["resume_epoch"]),  # TODO
    )
    _, _, _, exp_run, _ = get_dataset(config, exp=exp)
    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")
    best_states = []
    with open(best_states_file, "r") as f:
        for line in f.readlines():
            best_states.append(int(line.replace("\n", "")))

    model = get_model(config, nr_labels=label_inf["label_nr"])

    agent_cdd = get_agent(config, model=model, label_names=label_inf["label_names"])

    state_old = best_states[0]
    agent_cdd.restore_state(exp_run.paths["states"], state_old)
    agent_cdd.model.finish()
    state_new = best_states[-1]
    agent_cdd.restore_state(exp_run.paths["states"], state_new)

    with torch.no_grad():
        for data in tqdm(train_dataloader[2], disable=True):
            # get data
            inputs, target = agent_seq.get_inputs_targets(data)

            # forward seq bottom feature

            outputs_new_seq = agent_seq.model.unet_new.bottom_block(agent_seq.model.unet_new.encoder(inputs)[1])
            outputs_old_seq = agent_seq.model.unet_old.bottom_block(agent_seq.model.unet_old.encoder(inputs)[1])

            # forward kd bottom feature
            outputs_new_kd = agent_kd.model.unet_new.bottom_block(agent_kd.model.unet_new.encoder(inputs)[1])
            outputs_old_kd = agent_kd.model.unet_old.bottom_block(agent_kd.model.unet_old.encoder(inputs)[1])

            # forward cdd bottom feature
            outputs_new_cdd = agent_cdd.model.unet_new.bottom_block(agent_cdd.model.unet_new.encoder(inputs)[1])
            outputs_old_cdd = agent_cdd.model.unet_old.bottom_block(agent_cdd.model.unet_old.encoder(inputs)[1])

            features_seq_new.append(outputs_new_seq.cpu().numpy())
            features_seq_old.append(outputs_old_seq.cpu().numpy())

            features_kd_new.append(outputs_new_kd.cpu().numpy())
            features_kd_old.append(outputs_old_kd.cpu().numpy())

            features_cdd_new.append(outputs_new_cdd.cpu().numpy())
            features_cdd_old.append(outputs_old_cdd.cpu().numpy())

    save_dir = "ablation_results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features_seq_old = np.concatenate(features_seq_old, axis=0)
    features_seq_old = features_seq_old.reshape(features_seq_old.shape[0], -1)
    features_seq_new = np.concatenate(features_seq_new, axis=0)
    features_seq_new = features_seq_new.reshape(features_seq_new.shape[0], -1)
    dists_seq = np.linalg.norm(features_seq_new - features_seq_old, axis=1)

    features_kd_old = np.concatenate(features_kd_old, axis=0)
    features_kd_old = features_kd_old.reshape(features_kd_old.shape[0], -1)
    features_kd_new = np.concatenate(features_kd_new, axis=0)
    features_kd_new = features_kd_new.reshape(features_kd_new.shape[0], -1)
    dists_kd = np.linalg.norm(features_kd_new - features_kd_old, axis=1)

    features_cdd_old = np.concatenate(features_cdd_old, axis=0)
    features_cdd_old = features_cdd_old.reshape(features_cdd_old.shape[0], -1)
    features_cdd_new = np.concatenate(features_cdd_new, axis=0)
    features_cdd_new = features_cdd_new.reshape(features_cdd_new.shape[0], -1)
    dists_cdd = np.linalg.norm(features_cdd_new - features_cdd_old, axis=1)

    # pca together and plot
    features = np.concatenate(
        [
            features_seq_old,
            features_seq_new,
            features_kd_old,
            features_kd_new,
            features_cdd_old,
            features_cdd_new,
        ],
        axis=0,
    )
    labels = np.concatenate(
        [
            np.zeros(len(features_seq_old)),
            np.ones(len(features_seq_new)),
            np.ones(len(features_kd_old)) * 2,
            np.ones(len(features_kd_new)) * 3,
            np.ones(len(features_cdd_old)) * 4,
            np.ones(len(features_cdd_new)) * 5,
        ],
        axis=0,
    )
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Get the indices for each label
    idx_seq_old = labels == 0
    idx_seq_new = labels == 1
    idx_kd_old = labels == 2
    idx_kd_new = labels == 3
    idx_cdd_old = labels == 4
    idx_cdd_new = labels == 5

    # Plot each group separately

    plt.scatter(
        features_pca[idx_seq_new, 0],
        features_pca[idx_seq_new, 1],
        c="Orange",
        label="SEQ-new",
        alpha=0.9,
        edgecolors="white",
    )

    plt.scatter(
        features_pca[idx_kd_new, 0] + 10.0,
        features_pca[idx_kd_new, 1] + 50.0,
        c="red",
        label="KD-new",
        alpha=0.9,
        edgecolors="white",
    )

    plt.scatter(
        features_pca[idx_cdd_new, 0],
        features_pca[idx_cdd_new, 1],
        c="green",
        label="Dom.-new",
        alpha=0.9,
        edgecolors="white",
    )

    plt.scatter(
        features_pca[idx_seq_old, 0],
        features_pca[idx_seq_old, 1],
        c="cyan",
        label="SEQ-old",
        alpha=0.9,
        edgecolors="white",
    )

    plt.scatter(
        features_pca[idx_kd_old, 0] + 10.0,
        features_pca[idx_kd_old, 1] + 50.0,
        c="purple",
        label="KD-old",
        alpha=0.9,
        edgecolors="white",
    )

    plt.scatter(
        features_pca[idx_cdd_old, 0],
        features_pca[idx_cdd_old, 1],
        c="blue",
        label="Dom.-old",
        alpha=0.9,
        edgecolors="white",
    )

    # Show the legend
    plt.legend(prop={"family": "Times New Roman", "size": 16}, loc="lower right", ncol=2)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dom_both.png"), dpi=600)
    plt.show()

    # plot dist
    plt.figure(figsize=(12, 5))
    plt.hist(dists_seq, bins=100, alpha=0.5, label="SEQ distance")
    plt.hist(dists_kd, bins=100, alpha=0.5, label="KD distance")
    plt.hist(dists_cdd, bins=100, alpha=0.5, label="Dom. distance")
    plt.legend(prop={"family": "Times New Roman", "size": 16}, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dom_distance_L2.png"), dpi=600)
    plt.show()
