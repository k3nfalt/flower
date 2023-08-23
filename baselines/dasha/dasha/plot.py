import argparse
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from omegaconf import OmegaConf

from dasha.strategy import CompressionAggregator


def moving_average(x, w):
    w = np.ones((w,)) / w
    return scipy.ndimage.convolve1d(x, w)


def plot(args) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex="row")
    paths = []
    for path in args.input_paths:
        if not os.path.exists(os.path.join(path, "history")):
            local_folders = os.listdir(path)
            for folder in local_folders:
                if os.path.isdir(os.path.join(path, folder)):
                    paths.append(os.path.join(path, folder))
        else:
            paths.append(path)
    for save_path in paths:
        with open(os.path.join(save_path, "history"), "rb") as f:
            history = pickle.load(f)
        with open(os.path.join(save_path, "config.yaml"), "r") as f:
            cfg = OmegaConf.load(f)
        metric = history.metrics_distributed[CompressionAggregator.RECEIVED_BYTES]
        received_bytes_rounds, received_bytes = list(zip(*metric))
        if args.metric == 'loss':
            rounds, losses = list(zip(*history.losses_distributed))
            axs.set_yscale('log')
        elif args.metric == CompressionAggregator.SQUARED_GRADIENT_NORM:
            metrics = history.metrics_distributed[CompressionAggregator.SQUARED_GRADIENT_NORM]
            rounds, losses = list(zip(*metrics))
            axs.set_yscale('log')
        elif args.metric == CompressionAggregator.ACCURACY:
            metrics = history.metrics_distributed[CompressionAggregator.ACCURACY]
            rounds, losses = list(zip(*metrics))
        np.testing.assert_array_equal(received_bytes_rounds, rounds)
        target = cfg.method.client._target_
        client = target.split(".")[-1]
        losses = np.asarray(losses)
        if args.smooth_plot is not None:
            losses = moving_average(losses, args.smooth_plot)
        axs.plot(np.asarray(received_bytes), losses,
                 label=f"{client}; Step size: {cfg.method.strategy.step_size}")
        axs.set_ylabel(args.metric)
        axs.set_xlabel("#bits / client")
        axs.legend(loc="upper left")
        fig.savefig(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Results of Experiments")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        help="Paths to the results of the experiments",
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to a saved plot",
        required=True
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='loss',
        help="Type of metric to plot",
    )
    parser.add_argument(
        "--smooth-plot",
        type=int,
        default=None,
        help="Smooth plot in a window",
    )
    args = parser.parse_args()
    plot(args)
