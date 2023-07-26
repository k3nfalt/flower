from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

import torch
from torch.utils.data import Dataset

import flwr as fl
from flwr.common.typing import NDArrays, Scalar
from logging import DEBUG


class DashaClient(fl.client.NumPyClient):  
    def __init__(
        self,
        function: torch.nn.Module,
        dataset: Dataset,
        device: torch.device
    ):
        self._function = function
        self._prepare_input(dataset, device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        parameters = [val.detach().cpu().numpy().flatten() for _, val in self._function.named_parameters()]
        return [np.concatenate(parameters)]

    def set_parameters(self, parameters: NDArrays) -> None:
        assert len(parameters) == 1
        parameters = parameters[0]
        state_dict = {}
        shift = 0
        for k, parameter_layer in self._function.named_parameters():
            numel = parameter_layer.numel()
            parameter = parameters[shift:shift + numel].reshape(parameter_layer.shape)
            state_dict[k] = torch.Tensor(parameter)
            shift += numel
        self._function.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self._function.zero_grad()
        function_value = self._function(self._features, self._targets)
        function_value.backward()
        gradients = np.concatenate([val.grad.cpu().numpy().flatten() for val in self._function.parameters()])
        return [gradients], len(self._targets), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss = self._function(self._features, self._targets)
        return float(loss), len(self._targets), {}

    def _prepare_input(self, dataset, device):
        self._features, self._targets = dataset[:]
        self._features = self._features.to(device)
        self._targets = self._targets.to(device)
