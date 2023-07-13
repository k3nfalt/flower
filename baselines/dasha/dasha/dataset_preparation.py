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

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf


class DatasetType(Enum):
    LIBSVM = 'libsvm'
    
    
class DatasetSplit(Enum):
    TRAIN = 'train'


def find_pre_downloaded_or_download(cfg: DictConfig) -> None:
    """ Finds a pre-downloaded dataset or downloads a new one

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    
    assert cfg.dataset.type == DatasetType.LIBSVM.value
    path_to_dataset = cfg.dataset.path_to_dataset
    assert path_to_dataset is not None
    dataset_name = cfg.dataset.dataset_name
    target_file = os.path.join(path_to_dataset, "{}_{}".format(dataset_name, DatasetSplit.TRAIN.value))
    if os.path.exists(target_file):
        return
    print("Downloading the dataset")
    dataset_url = cfg.dataset._dataset_urls[dataset_name][DatasetSplit.TRAIN.value]
    urllib.request.urlretrieve(dataset_url, target_file)
