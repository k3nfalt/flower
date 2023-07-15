from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import Dataset

import flwr as fl
from flwr.common.typing import NDArrays, Scalar


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
        return [val.detach().cpu().numpy() for _, val in self._function.named_parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        state_dict = {k: torch.Tensor(parameters[i]) for i, (k, _) in enumerate(self._function.named_parameters())}
        self._function.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self._function.zero_grad()
        function_value = self._function(self._features, self._targets)
        function_value.backward()
        gradients = [val.grad.cpu().numpy() for val in self._function.parameters()]
        return gradients, len(self._targets), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss = self._function(self._features, self._targets)
        return float(loss), len(self._targets), {}

    def _prepare_input(self, dataset, device):
        self._features, self._targets = dataset[:]
        self._features = self._features.to(device)
        self._targets = self._targets.to(device)
