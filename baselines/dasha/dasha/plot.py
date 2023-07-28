import argparse
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf


def plot(args) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex="row")
    paths = []
    for path in args.input_paths:
        if os.path.exists(os.path.join(path, "multirun.yaml")):
            local_folders = os.listdir(path)
            for folder in local_folders:
                if folder != "multirun.yaml":
                    paths.append(os.path.join(path, folder))
        else:
            paths.append(path)
    for save_path in paths:
        with open(os.path.join(save_path, "history"), "rb") as f:
            history = pickle.load(f)
        with open(os.path.join(save_path, "config.yaml"), "r") as f:
            cfg = OmegaConf.load(f)
        rounds, losses = list(zip(*history.losses_distributed))
        target = cfg.client._target_
        client = target.split(".")[-1]
        axs.plot(np.asarray(rounds), np.asarray(losses), 
                 label=f"{client}; Step size: {cfg.strategy.step_size}")
        axs.set_ylabel("Loss")
        axs.set_xlabel("Rounds")
        axs.legend(loc="upper left")
        fig.savefig(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Results of Experiments")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        help="Paths to the results of the experiments",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to a saved plot",
    )
    args = parser.parse_args()
    plot(args)
