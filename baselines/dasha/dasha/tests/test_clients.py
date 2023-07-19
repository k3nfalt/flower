import unittest

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np

from dasha.client import DashaClient


_CPU_DEVICE = "cpu"


class DummyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.Tensor([2]))

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean((self._weight.view(-1, 1) * features - targets)**2)


class TestDashaClient(unittest.TestCase):
    def setUp(self) -> None:
        self._function = DummyNet()
        self._features = [[1], [2]]
        self._targets = [[1], [2]]
        dataset = data_utils.TensorDataset(torch.Tensor(self._features), 
                                           torch.Tensor(self._targets))
        self._client = DashaClient(function=self._function, 
                                   dataset=dataset,
                                   device=_CPU_DEVICE)

    def testGetParameters(self) -> None:
        parameters = self._client.get_parameters(config={})
        self.assertEqual(len(parameters), 1)
        self.assertAlmostEqual(float(parameters[0]), 2)
    
    def testSetParameters(self) -> None:
        parameter = 3.0
        parameters = [np.array([parameter])]
        self._client.set_parameters(parameters)
        self.assertAlmostEqual(float(self._function._weight.detach().numpy()), parameter)

    def testEvaluate(self) -> None:
        parameter = 3.0
        parameters_list = [np.array([parameter])]
        loss, num_samples, _ = self._client.evaluate(parameters_list, config={})
        self.assertEqual(num_samples, 2)
        loss_actual = sum([0.5 * (parameter * self._features[i][0] - self._targets[i][0]) ** 2 
                           for i in range(len(self._targets))]) / len(self._targets)
        self.assertAlmostEqual(float(loss), loss_actual)

    def testFit(self) -> None:
        parameter = 3.0
        parameters_list = [np.array([parameter])]
        gradients, num_samples, _ = self._client.fit(parameters_list, config={})
        self.assertEqual(num_samples, 2)
        self.assertEqual(len(gradients), 1)
        gradient_actual = sum([self._features[i][0] * (parameter * self._features[i][0] - self._targets[i][0])
                           for i in range(len(self._targets))]) / len(self._targets)
        self.assertAlmostEqual(float(gradients[0]), gradient_actual)


if __name__ == "__main__":
    unittest.main()
