import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING

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

from dasha.compressors import decompress, IdentityUnbiasedCompressor
from dasha.client import DashaClient, MarinaClient


class CompressionAggregator(Strategy):
    _EMPTY_CONFIG = {}
    _SKIPPED = 'skipped'
    def __init__(self, step_size, num_clients):
        self._step_size = step_size
        self._parameters = None
        self._gradient_estimator = None
        self._num_clients = num_clients
    
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
        return loss_aggregated, {}

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
    
    def _bernoulli_sample(self, random_generator, prob):
        if prob == 0.0:
            return False
        return random_generator.random() < prob
