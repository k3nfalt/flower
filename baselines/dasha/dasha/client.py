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
from dasha.models import ClassificationModel


class CompressionClient(fl.client.NumPyClient):
    _SEND_FULL_GRADIENT = 'send_full_gradient'
    ACCURACY = 'accuracy'
    GRADIENT = 'gradient'
    SIZE_OF_COMPRESSED_VECTORS = 'size_of_compressed_vectors'
    def __init__(
        self,
        function: ClassificationModel,
        dataset: Dataset,
        device: torch.device,
        compressor: Optional[UnbiasedBaseCompressor] = None,
        evaluate_accuracy=False
    ):
        self._function = function
        self._compressor = compressor if compressor is not None else IdentityUnbiasedCompressor()
        self._local_gradient_estimator = None
        self._gradient_estimator = None
        self._momentum = None
        self._evaluate_accuracy = evaluate_accuracy
        self._dataset = dataset
        self._device = device

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        parameters = [val.detach().cpu().numpy().flatten() for _, val in self._function.named_parameters()]
        return [np.concatenate(parameters)]

    def _set_parameters(self, parameters: NDArrays) -> None:
        assert len(parameters) == 1
        parameters = parameters[0]
        self._compressor.set_dim(len(parameters))
        state_dict = {}
        shift = 0
        for k, parameter_layer in self._function.named_parameters():
            numel = parameter_layer.numel()
            parameter = parameters[shift:shift + numel].reshape(parameter_layer.shape)
            state_dict[k] = torch.Tensor(parameter)
            shift += numel
        self._function.load_state_dict(state_dict, strict=True)
    
    def _get_current_gradients(self):
        return np.concatenate([val.grad.cpu().numpy().flatten() for val in self._function.parameters()])


class GradientCompressionClient(CompressionClient):
    def __init__(self, *args, send_gradient=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._send_gradient = send_gradient
        self._features, self._targets = None, None
        self._prepare_input(self._dataset, self._device)
        
    def _prepare_input(self, dataset, device):
        self._features, self._targets = dataset[:]
        self._features = self._features.to(device)
        self._targets = self._targets.to(device)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        if config[self._SEND_FULL_GRADIENT]:
            compressed_gradient = self._gradient_step(parameters)
        else:
            compressed_gradient = self._compression_step(parameters)
        return compressed_gradient, len(self._targets), {self.SIZE_OF_COMPRESSED_VECTORS: self._compressor.num_nonzero_components()}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self._set_parameters(parameters)
        loss = self._function(self._features, self._targets)
        metrics = {}
        if self._send_gradient:
            loss.backward()
            gradients = self._get_current_gradients()
            metrics["gradient"] = gradients.astype(np.float32).tobytes()
        if self._evaluate_accuracy:
            accuracy = self._function.accuracy(self._features, self._targets)
            metrics[self.ACCURACY] = accuracy
        return float(loss), len(self._targets), metrics
    
    def _calculate_gradient(self, parameters: NDArrays):
        self._set_parameters(parameters)
        self._function.zero_grad()
        loss = self._function(self._features, self._targets)
        loss.backward()
        gradients = self._get_current_gradients()
        return gradients
    
    def _gradient_step(self):
        raise NotImplementedError()
    
    def _compression_step(self):
        raise NotImplementedError()


class DashaClient(GradientCompressionClient):
    def _gradient_step(self, parameters: NDArrays):
        gradients = self._calculate_gradient(parameters)
        self._gradient_estimator = gradients
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(self._gradient_estimator)
        return compressed_gradient
    
    def _compression_step(self, parameters: NDArrays):
        gradients = self._calculate_gradient(parameters)
        momentum = self._get_momentum()
        compressed_gradient = self._compressor.compress(
            gradients - self._local_gradient_estimator - momentum * (self._gradient_estimator - self._local_gradient_estimator))
        self._local_gradient_estimator = gradients
        self._gradient_estimator += decompress(compressed_gradient)
        return compressed_gradient
    
    def _get_momentum(self):
        if self._momentum is not None:
            return self._momentum
        self._momentum = 1 / (1 + 2 * self._compressor.omega())
        return self._momentum


class MarinaClient(GradientCompressionClient):
    def _gradient_step(self, parameters: NDArrays):
        gradients = self._calculate_gradient(parameters)
        assert self._gradient_estimator is None
        self._local_gradient_estimator = gradients
        compressed_gradient = IdentityUnbiasedCompressor().compress(gradients)
        return compressed_gradient
    
    def _compression_step(self, parameters: NDArrays):
        gradients = self._calculate_gradient(parameters)
        assert self._gradient_estimator is None
        compressed_gradient = self._compressor.compress(gradients - self._local_gradient_estimator)
        self._local_gradient_estimator = gradients
        return compressed_gradient


class StochasticGradientCompressionClient(CompressionClient):
    _LARGE_NUMBER = 10**12
    def __init__(self, *args, batch_size=1, num_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._previous_parameters = None
        self._batch_sampler = iter(torch.utils.data.DataLoader(
            self._dataset, batch_size=self._batch_size, num_workers=num_workers, 
            sampler=torch.utils.data.RandomSampler(self._dataset, replacement=True, num_samples=self._LARGE_NUMBER)))
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        if config[self._SEND_FULL_GRADIENT]:
            compressed_gradient = self._stochastic_gradient_step(parameters)
        else:
            compressed_gradient = self._stochastic_compression_step(parameters)
        return compressed_gradient, len(self._targets), {self.SIZE_OF_COMPRESSED_VECTORS: self._compressor.num_nonzero_components()}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self._set_parameters(parameters)
        # loss = self._function(self._features, self._targets)
        metrics = {}
        if self._evaluate_accuracy:
            accuracy = self._function.accuracy(self._features, self._targets)
            metrics[self.ACCURACY] = accuracy
        return float(loss), len(self._targets), metrics
    
    def _calculate_stochastic_gradient(self, parameters: NDArrays):
        features, labels = next(self._batch_sampler)
        self._set_parameters(parameters)
        self._function.zero_grad()
        loss = self._function(self._features, self._targets)
        loss.backward()
        gradients = self._get_current_gradients()
        return gradients
    
    def _stochastic_gradient_step(self):
        raise NotImplementedError()
    
    def _stochastic_compression_step(self):
        raise NotImplementedError()