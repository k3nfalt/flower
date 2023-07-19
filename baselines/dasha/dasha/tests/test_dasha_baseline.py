import os
import unittest

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np

from omegaconf import OmegaConf


from dasha.dataset import LIBSVMDatasetName
from dasha.dataset_preparation import DatasetType
from dasha.main import run_parallel


TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


class TestDashaBaseline(unittest.TestCase):
    def testBaseline(self) -> None:
        cfg = OmegaConf.create({
            "dataset": {
                "type": DatasetType.LIBSVM.value,
                "path_to_dataset": TESTDATA_PATH,
                "dataset_name": LIBSVMDatasetName.MUSHROOMS.value
            },
            "num_clients": 4,
            "num_rounds": 10,
            "strategy": {
                "_target_": "dasha.strategy.DashaStrategy",
                "step_size": 0.01
            }
        })
        run_parallel(cfg)


if __name__ == "__main__":
    unittest.main()
