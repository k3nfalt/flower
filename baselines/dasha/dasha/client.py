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

from dasha.compressors import UnbiasedBaseCompressor, IdentityUnbiasedCompressor, decompress


class CompressionClient(fl.client.NumPyClient):
    _SEND_FULL_GRADIENT = 'send_full_gradient'
    def __init__(
        self,
        function: torch.nn.Module,
        dataset: Dataset,
        device: torch.device,
        compressor: UnbiasedBaseCompressor = IdentityUnbiasedCompressor()
    ):
        self._function = function
        self._compressor = compressor
        self._local_gradient_estimator = None
        self._gradient_estimator = None
        self._momentum = None
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
        if config[self._SEND_FULL_GRADIENT]:
            compressed_gradient = self._gradient_step(gradients)
        else:
            compressed_gradient = self._compression_step(gradients)
        return compressed_gradient, len(self._targets), {}
    
    def _get_momentum(self, dim):
        if self._momentum is not None:
            return self._momentum
        self._compressor.set_dim(dim)
        self._momentum = 1 / (1 + 2 * self._compressor.omega())
        return self._momentum

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss = self._function(self._features, self._targets)
        return float(loss), len(self._targets), {}

    def _prepare_input(self, dataset, device):
        self._features, self._targets = dataset[:]
        self._features = self._features.to(device)
        self._targets = self._targets.to(device)
        
    def _gradient_step(self, gradients):
        raise NotImplementedError()
    
    def _compression_step(self, gradients):
        raise NotImplementedError()


class DashaClient(CompressionClient):
    def _gradient_step(self, gradients):
        assert self._gradient_estimator is None
        self._gradient_estimator = gradients
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(self._gradient_estimator)
        return compressed_gradient
    
    def _compression_step(self, gradients):
        momentum = self._get_momentum(len(gradients))
        compressed_gradient = self._compressor.compress(
            gradients - self._local_gradient_estimator - momentum * (self._gradient_estimator - self._local_gradient_estimator))
        self._local_gradient_estimator = gradients
        self._gradient_estimator += decompress(compressed_gradient)
        return compressed_gradient


class MarinaClient(CompressionClient):
    def _gradient_step(self, gradients):
        assert self._gradient_estimator is None
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(self._gradient_estimator)
        return compressed_gradient
    
    def _compression_step(self, gradients):
        assert self._gradient_estimator is None
        compressed_gradient = self._compressor.compress(gradients - self._local_gradient_estimator)
        self._local_gradient_estimator = gradients
        return compressed_gradient
