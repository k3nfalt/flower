import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING, INFO

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log

from dasha.compressors import decompress, IdentityUnbiasedCompressor, estimate_size
from dasha.client import DashaClient, MarinaClient, CompressionClient


class CompressionAggregator(Strategy):
    _EMPTY_CONFIG = {}
    _SKIPPED = 'skipped'
    SQUARED_GRADIENT_NORM = 'squared_gradient_norm'
    RECEIVED_BYTES = 'received_bytes'
    def __init__(self, step_size, num_clients):
        self._step_size = step_size
        self._parameters = None
        self._gradient_estimator = None
        self._num_clients = num_clients
        self._total_received_bytes_per_client_during_fit = 0
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        raise NotImplementedError()

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evel_ins = EvaluateIns(parameters, self._EMPTY_CONFIG)
        return [(client, evel_ins) for client in client_manager.all().values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        assert len(failures) == 0, failures
        if len(results) != self._num_clients:
            log(WARNING, "not all clients have sent results. Waiting and repeating...")
            time.sleep(1.0)
            return ndarrays_to_parameters([self._parameters]), {}
        parsed_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        expect_compressor = IdentityUnbiasedCompressor.name() if self._gradient_estimator is None else None
        estimated_sizes = [estimate_size(compressed_params) for compressed_params in parsed_results]
        max_estimated_size = int(np.max(estimated_sizes))
        self._total_received_bytes_per_client_during_fit += max_estimated_size
        parsed_results = [decompress(compressed_params, assert_compressor=expect_compressor) 
                          for compressed_params in parsed_results]
        aggregated_vector = sum(parsed_results) / len(parsed_results)
        if self._gradient_estimator is None:
            self._gradient_estimator = aggregated_vector
            self._parameters -= self._step_size * self._gradient_estimator
            return ndarrays_to_parameters([self._parameters]), {}
        self._gradient_estimator += aggregated_vector
        self._parameters -= self._step_size * self._gradient_estimator
        return ndarrays_to_parameters([self._parameters]), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        assert len(failures) == 0
        loss_aggregated = weighted_loss_avg([(1, evaluate_res.loss) for _, evaluate_res in results])
        log(INFO, "Round: {}".format(server_round))
        log(INFO, "Aggregated loss: {}".format(loss_aggregated))
        metrics = {self.RECEIVED_BYTES: self._total_received_bytes_per_client_during_fit}
        if CompressionClient.GRADIENT in results[0][1].metrics:
            gradients = [evaluate_res.metrics[CompressionClient.GRADIENT] for _, evaluate_res in results]
            gradients = [np.frombuffer(gradient, dtype=np.float32) for gradient in gradients]
            gradient = sum(gradients) / len(gradients)
            norm_square = float(np.linalg.norm(gradient) ** 2)
            metrics[self.SQUARED_GRADIENT_NORM] = norm_square
            log(INFO, "Squared gradient norm: {}".format(norm_square))
        if CompressionClient.ACCURACY in results[0][1].metrics:
            accuracy_aggregated = weighted_loss_avg([(1, evaluate_res.metrics[CompressionClient.ACCURACY]) 
                                                     for _, evaluate_res in results])
            metrics[CompressionClient.ACCURACY] = accuracy_aggregated
            log(INFO, "Aggregated accuracy: {}".format(accuracy_aggregated))
        return loss_aggregated, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round == 0:
            parameters = parameters_to_ndarrays(parameters)
            assert len(parameters) == 1
            self._parameters = parameters[0]
        return None


class DashaAggregator(CompressionAggregator):
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, {DashaClient._SEND_FULL_GRADIENT: self._gradient_estimator is None})
        return [(client, fit_ins) for client in client_manager.all().values()]


class MarinaAggregator(CompressionAggregator):
    def __init__(self, *args, seed=None, size_of_compressed_vectors=None, **kwargs):
        super(MarinaAggregator, self).__init__(*args, **kwargs)
        self._generator = np.random.default_rng(seed)
        assert size_of_compressed_vectors is not None
        self._size_of_compressed_vectors = size_of_compressed_vectors
        self._prob = None
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        prob = self._get_prob()
        self._gradient_estimator = None if self._bernoulli_sample(self._generator, prob) else self._gradient_estimator
        fit_ins = FitIns(parameters, {MarinaClient._SEND_FULL_GRADIENT: self._gradient_estimator is None})
        return [(client, fit_ins) for client in client_manager.all().values()]
    
    def _get_prob(self):
        if self._prob is not None:
            return self._prob
        self._prob = self._size_of_compressed_vectors / len(self._parameters)
        return self._prob
    
    def _bernoulli_sample(self, random_generator, prob):
        if prob == 0.0:
            return False
        return random_generator.random() < prob
