import sys
import traceback
import time
import os
import pickle

from typing import Tuple
# import concurrent.futures
from multiprocessing import Pool

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np

import flwr as fl

import dasha.dataset
from dasha.dataset_preparation import find_pre_downloaded_or_download_dataset


LOCAL_ADDRESS = "localhost:8080"


def _generate_seed(generator):
    return generator.integers(10e9)


def _parallel_run(params: Tuple[DictConfig, int, int]) -> None:
    try:
        cfg, index_parallel, seed = params
        if index_parallel == 0:
            strategy_instance = instantiate(cfg.strategy, num_clients=cfg.num_clients)
            return fl.server.start_server(server_address=LOCAL_ADDRESS, 
                                          config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                                          strategy=strategy_instance)
        else:
            index_client = index_parallel - 1
            dataset = dasha.dataset.load_dataset(cfg)
            datasets = dasha.dataset.random_split(dataset, cfg.num_clients)
            local_dataset = datasets[index_client]
            function = instantiate(cfg.model)
            compressor = instantiate(cfg.compressor, seed=seed)
            client_instance = instantiate(cfg.client, 
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
    with Pool(processes=cfg.num_clients + 1) as pool:
        seed = _generate_seed(generator)
        results = pool.map(_parallel_run, [(cfg, index_parallel, seed) for index_parallel in range(cfg.num_clients + 1)])
    return results[0]


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    find_pre_downloaded_or_download_dataset(cfg)
    history = run_parallel(cfg)
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


if __name__ == "__main__":
    main()
