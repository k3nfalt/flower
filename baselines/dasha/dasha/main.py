import sys
import traceback
import time

from typing import Tuple
# import concurrent.futures
from multiprocessing import Pool

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

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
    run_parallel(cfg)
    
    # dataloaders = load_dataset(cfg)
    
    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()
    # client_fn = client.gen_client_fn(
    #     model=cfg.model,
    # )

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy_instance = instantiate(cfg.strategy)

    # 5. Start Simulation
    # history = fl.simulation.start_simulation(
    #     client_fn=client_fn,
    #     num_clients=cfg.num_clients,
    #     config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
    #     client_resources={
    #         "num_cpus": cfg.client_resources.num_cpus,
    #         "num_gpus": cfg.client_resources.num_gpus,
    #     },
    #     strategy=strategy_instance,
    # )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir
    
if __name__ == "__main__":
    main()