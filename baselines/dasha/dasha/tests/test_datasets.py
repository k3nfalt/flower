import os
import unittest

import numpy as np

from omegaconf import OmegaConf

from dasha.dataset import load_dataset, LIBSVMDatasetName
from dasha.dataset_preparation import DatasetType


TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


class TestLoadMushroomsDataset(unittest.TestCase):
    def test(self) -> None:
        cfg = OmegaConf.create({"dataset": {
            "type": DatasetType.LIBSVM.value,
            "path_to_dataset": TESTDATA_PATH,
            "dataset_name": LIBSVMDatasetName.MUSHROOMS.value
        }})
        dataset = load_dataset(cfg)
        features, labels = dataset[:]
        self.assertEqual(np.sort(np.unique(labels.numpy())).tolist(), [0, 1])
        self.assertEqual(list(features.shape), [8124, 112])


if __name__ == "__main__":
    unittest.main()
