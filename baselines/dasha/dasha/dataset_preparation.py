"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import os
import urllib.request
from enum import Enum
from typing import Dict
import tempfile

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

import torchvision


class DatasetType(Enum):
    LIBSVM = 'libsvm'
    CIFAR10 = 'cifar10'
    TEST = 'test'
    RANDOM_TEST = 'random_test'
    
    
class DatasetSplit(Enum):
    TRAIN = 'train'


def train_dataset_path(path_to_dataset, dataset_name):
    return os.path.join(path_to_dataset, "{}_{}".format(dataset_name, DatasetSplit.TRAIN.value))


def _prepare_libsvm(path_to_dataset: str, dataset_name: str, dataset_urls: Dict[str, str]) -> None:
    assert path_to_dataset is not None
    target_file = train_dataset_path(path_to_dataset, dataset_name)
    if os.path.exists(target_file):
        return
    print("Downloading the dataset")
    dataset_url = dataset_urls[dataset_name][DatasetSplit.TRAIN.value]
    urllib.request.urlretrieve(dataset_url, target_file)


def _prepare_cifar10(path_to_dataset: str) -> None:
    torchvision.datasets.CIFAR10(root=path_to_dataset, train=True, download=True)


def find_pre_downloaded_or_download_dataset(cfg: DictConfig) -> None:
    """ Finds a pre-downloaded dataset or downloads it

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    if cfg.dataset.path_to_dataset is None:
        path_to_dataset = tempfile.mkdtemp()
        print(f"The parameter cfg.dataset.path_to_dataset is not specified. Creating a temporary directory in {path_to_dataset}.")
        cfg.dataset.path_to_dataset = path_to_dataset
    if cfg.dataset.type == DatasetType.LIBSVM.value:
        _prepare_libsvm(cfg.dataset.path_to_dataset, cfg.dataset.dataset_name, cfg.dataset._dataset_urls)
    elif cfg.dataset.type == DatasetType.CIFAR10.value:
        _prepare_cifar10(cfg.dataset.path_to_dataset)
    else:
        raise RuntimeError()
