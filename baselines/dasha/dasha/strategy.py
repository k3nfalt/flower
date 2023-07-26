import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING

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

from dasha.compressors import decompress


class DashaStrategy(Strategy):
    _EMPTY_CONFIG = {}
    _SKIPPED = 'skipped'
    def __init__(self, step_size, num_clients):
        self._step_size = step_size
        self._parameters = None
        self._gradient_estimators = None
        self._num_clients = num_clients
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, self._EMPTY_CONFIG)
        return [(client, fit_ins) for client in client_manager.all().values()]

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
            log(WARNING, "Warning: not all clients have sent results. Waiting and repeating...")
            time.sleep(1.0)
            return ndarrays_to_parameters([self._parameters]), {}
        parsed_results = [(parameters_to_ndarrays(fit_res.parameters), 1) for _, fit_res in results]
        parsed_results = [(decompress(*compressed_params), weight) for compressed_params, weight in parsed_results]
        aggregated_vectors = aggregate(parsed_results)
        assert len(aggregated_vectors) == 1
        aggregated_vector = aggregated_vectors[0]
        gradient_estimator = aggregated_vector
        self._parameters -= self._step_size * gradient_estimator
        return ndarrays_to_parameters([self._parameters]), {}
        
        # for parameter, gradient_estimator in zip(self._parameters, self._gradient_estimators):
        #     parameter -= self._step_size * gradient_estimator
        # assert len(failures) == 0
        # parsed_results = [(parameters_to_ndarrays(fit_res.parameters), 1) for _, fit_res in results]
        # aggregated_vectors = aggregate(parsed_results)
        # for aggregated_vector, gradient_estimator in zip(aggregated_vectors, self._gradient_estimators):
        #     gradient_estimator += aggregated_vector
        # return ndarrays_to_parameters(self._parameters), {}

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
