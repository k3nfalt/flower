from enum import Enum

import torch
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from omegaconf import DictConfig
from sklearn.datasets import load_svmlight_file
import numpy as np

from dasha.dataset_preparation import DatasetType, train_dataset_path


class LIBSVMDatasetName(Enum):
    MUSHROOMS = 'mushrooms'


def load_libsvm_dataset(cfg: DictConfig) -> Dataset:
    assert cfg.dataset.type == DatasetType.LIBSVM.value
    path_to_dataset = cfg.dataset.path_to_dataset
    dataset_name = cfg.dataset.dataset_name
    data, labels = load_svmlight_file(train_dataset_path(path_to_dataset, dataset_name))
    data = data.toarray().astype(np.float32)
    print("Original labels: {}".format(np.unique(labels, return_counts=True)))
    print("Features Shape: {}".format(data.shape))
    if dataset_name == LIBSVMDatasetName.MUSHROOMS.value:
        labels = labels.astype(np.int64) - 1
    else:
        raise RuntimeError("Wrong dataset")
    dataset = data_utils.TensorDataset(torch.Tensor(data), torch.Tensor(labels))
    return dataset


def load_test_dataset(cfg: DictConfig) -> Dataset:
    assert cfg.dataset.type == DatasetType.TEST.value
    features = [[1], [2]]
    targets = [[1], [2]]
    dataset = data_utils.TensorDataset(torch.Tensor(features), torch.Tensor(targets))
    return dataset


def load_dataset(cfg: DictConfig) -> Dataset:
    if cfg.dataset.type == DatasetType.LIBSVM.value:
        return load_libsvm_dataset(cfg)
    elif cfg.dataset.type == DatasetType.TEST.value:
        return load_test_dataset(cfg)
    else:
        raise RuntimeError("Wrong dataset type")


def random_split(dataset, num_clients, seed=42):
    lengths = [1 / num_clients] * num_clients
    datasets = torch.utils.data.random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return datasets