import os
import unittest
import multiprocessing

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np

from omegaconf import OmegaConf


from dasha.dataset_preparation import DatasetType
from dasha.main import run_parallel
from dasha.tests.test_clients import DummyNetTwoParameters
from dasha.dataset import load_test_dataset


TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


def gradient_descent(step_size, num_rounds):
    dummy_net = DummyNetTwoParameters([1])
    dataset = load_test_dataset(
        OmegaConf.create({
            "dataset": {
                "type": DatasetType.TEST.value,
            }}))
    features, labels = dataset[:]
    results = []
    for _ in range(num_rounds):
        dummy_net.zero_grad()
        loss = dummy_net(features, labels)
        loss.backward()
        for weight in dummy_net.parameters():
            weight.data.sub_(step_size * weight.grad)
        results.append(float(dummy_net(features, labels).detach().numpy()))
    return results


class TestDashaBaseline(unittest.TestCase):
    def testBaseline(self) -> None:
        step_size = 0.1
        num_rounds = 10
        reference_results = gradient_descent(step_size, num_rounds)
        
        cfg = OmegaConf.create({
            "dataset": {
                "type": DatasetType.TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "compressor": {
                "_target_": "dasha.compressors.IdentityUnbiasedCompressor",
            },
            "model": {
                "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
            },
            "method": {
                "strategy": {
                    "_target_": "dasha.strategy.DashaAggregator",
                    "step_size": step_size
                },
                "client": {
                    "_target_": "dasha.client.DashaClient",
                    "device": "cpu"
                }
            }
        })
        results = run_parallel(cfg)
        results = [loss for (_, loss) in results.losses_distributed]
        # TODO: Maybe fix it. I don't know in which round Flower will start training in advance, so I check different subarrays for equality.
        self.assertTrue(np.any([np.allclose(reference_results[:len(results)-i], results[i:]) for i in range(2)]))


class TestDashaBaselineWithRandK(unittest.TestCase):
    def testBaseline(self) -> None:
        step_size = 0.01
        num_rounds = 100
        
        cfg = OmegaConf.create({
            "dataset": {
                "type": DatasetType.TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "model": {
                "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
            },
            "compressor": {
                "_target_": "dasha.compressors.RandKCompressor",
                "number_of_coordinates": 1
            },
            "method": {
                "strategy": {
                    "_target_": "dasha.strategy.DashaAggregator",
                    "step_size": step_size
                },
                "client": {
                    "_target_": "dasha.client.DashaClient",
                    "device": "cpu"
                }
            }
        })
        results = run_parallel(cfg)
        results = [loss for (_, loss) in results.losses_distributed]
        self.assertGreater(results[0], 1.0)
        self.assertLess(results[-1], 1e-5)


class TestMarinaBaselineWithRandK(unittest.TestCase):
    def testBaseline(self) -> None:
        step_size = 0.01
        num_rounds = 100
        number_of_coordinates = 1
        
        cfg = OmegaConf.create({
            "dataset": {
                "type": DatasetType.TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "model": {
                "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
            },
            "compressor": {
                "_target_": "dasha.compressors.RandKCompressor",
                "number_of_coordinates": number_of_coordinates
            },
            "method": {
                "strategy": {
                    "_target_": "dasha.strategy.MarinaAggregator",
                    "step_size": step_size,
                    "size_of_compressed_vectors": number_of_coordinates
                },
                "client": {
                    "_target_": "dasha.client.MarinaClient",
                    "device": "cpu"
                }
            }
        })
        results = run_parallel(cfg)
        results = [loss for (_, loss) in results.losses_distributed]
        self.assertGreater(results[0], 1.0)
        self.assertLess(results[-1], 1e-5)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')
    unittest.main()
