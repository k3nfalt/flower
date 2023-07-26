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
    dummy_net = DummyNetTwoParameters()
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
        multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        step_size = 0.1
        num_rounds = 10
        reference_results = gradient_descent(step_size, num_rounds)
        
        cfg = OmegaConf.create({
            "dataset": {
                "type": DatasetType.TEST.value,
            },
            "num_clients": 2,
            "num_rounds": num_rounds,
            "strategy": {
                "_target_": "dasha.strategy.DashaStrategy",
                "step_size": step_size
            },
            "model": {
                "_target_": "dasha.tests.test_clients.DummyNetTwoParameters",
            },
            "client": {
                "_target_": "dasha.client.DashaClient",
                "device": "cpu"
            }
        })
        results = run_parallel(cfg)
        results = [loss for (_, loss) in results.losses_distributed]
        # TODO: Maybe fix it. I don't know in which round Flower will start training in advance, so I check different subarrays for equality.
        self.assertTrue(np.any([np.allclose(reference_results[:len(results)-i], results[i:]) for i in range(2)]))


if __name__ == "__main__":
    unittest.main()
