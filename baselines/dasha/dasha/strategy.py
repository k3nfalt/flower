from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class DashaStrategy(fl.server.strategy.Strategy):
    _EMPTY_CONFIG = {}
    def __init__(self):
        pass
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, self._EMPTY_CONFIG)
        return [(client, fit_ins) for client in client_manager.all()]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        assert len(failures) == 0
        parsed_results = [(parameters_to_ndarrays(fit_res.parameters), 1)
                          for _, fit_res in results]
        aggregated_vectors = ndarrays_to_parameters(aggregate(parsed_results))
        return aggregated_vectors, {}

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
        return None
