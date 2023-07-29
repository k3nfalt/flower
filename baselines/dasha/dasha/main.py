import sys
import traceback
import time
import os
import pickle

from typing import Tuple, Optional
# import concurrent.futures
# from multiprocessing import Pool
import multiprocessing

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np

import flwr as fl
from flwr.server.history import History

import dasha.dataset
from dasha.dataset_preparation import find_pre_downloaded_or_download_dataset


LOCAL_ADDRESS = "localhost:8080"


def _generate_seed(generator):
    return generator.integers(10e9)


def _get_dataset_input_shape(dataset):
    assert len(dataset) > 0
    sample_features, _ = dataset[0]
    return list(sample_features.shape)


def save_history(history, cfg: DictConfig):
    if cfg.save_path is not None:
        assert not os.path.exists(cfg.save_path)
        os.mkdir(cfg.save_path)
        save_path = cfg.save_path
    else:
        save_path = HydraConfig.get().runtime.output_dir
    print(f"Saving to {save_path}")
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(os.path.join(save_path, "history"), "wb") as f:
        pickle.dump(history, f)


def _parallel_run(cfg: DictConfig, index_parallel: int, seed: int, queue: multiprocessing.Queue) -> None:
    try:
        if index_parallel == 0:
            strategy_instance = instantiate(cfg.method.strategy, num_clients=cfg.num_clients)
            history = fl.server.start_server(server_address=LOCAL_ADDRESS, 
                                             config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                                             strategy=strategy_instance)
            queue.put(history)
        else:
            index_client = index_parallel - 1
            dataset = dasha.dataset.load_dataset(cfg)
            datasets = dasha.dataset.random_split(dataset, cfg.num_clients)
            local_dataset = datasets[index_client]
            function = instantiate(cfg.model, input_shape=_get_dataset_input_shape(dataset))
            compressor = instantiate(cfg.compressor, seed=seed)
            client_instance = instantiate(cfg.method.client, 
                                          function=function,
                                          dataset=local_dataset,
                                          compressor=compressor)
            # TODO: Fix it
            time.sleep(1.0)
            fl.client.start_numpy_client(server_address=LOCAL_ADDRESS, 
                                         client=client_instance)
    except Exception as ex:
        print(traceback.format_exc())


def run_parallel(cfg: DictConfig) -> None:
    sys.stderr = sys.stdout
    generator = np.random.default_rng(seed=42)
    processes = []
    queue = multiprocessing.Queue()
    for index_parallel in range(cfg.num_clients + 1):
        seed = _generate_seed(generator)
        process = multiprocessing.Process(target=_parallel_run, args=(cfg, index_parallel, seed, queue))
        process.start()
        processes.append(process)
    history = queue.get()
    for process in processes:
        process.join()
    return history


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    find_pre_downloaded_or_download_dataset(cfg)
    history = run_parallel(cfg)
    save_history(history, cfg)


if __name__ == "__main__":
    main()
