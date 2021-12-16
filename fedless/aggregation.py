import abc
import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np
import pymongo
import tensorflow as tf

from fedless.data import DatasetLoaderBuilder
from fedless.models import (
    Parameters,
    ClientResult,
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    AggregatorFunctionResult,
    SerializedParameters,
    TestMetrics,
    DatasetLoaderConfig,
    ModelLoaderConfig,
    SimpleModelLoaderConfig,
    SerializedModel,
)
from fedless.persistence import (
    ClientResultDao,
    ParameterDao,
    PersistenceError,
    ModelDao,
)
from fedless.serialization import (
    deserialize_parameters,
    WeightsSerializerBuilder,
    SerializationError,
)

logger = logging.getLogger(__name__)


class AggregationError(Exception):
    pass


class InsufficientClientResults(AggregationError):
    pass


class UnknownCardinalityError(AggregationError):
    pass


class InvalidParameterShapeError(AggregationError):
    pass


def default_aggregation_handler(
    session_id: str,
    round_id: int,
    database: MongodbConnectionConfig,
    serializer: WeightsSerializerConfig,
    online: bool = False,
    test_data: Optional[DatasetLoaderConfig] = None,
    test_batch_size: int = 512,
    delete_results_after_finish: bool = True,
) -> AggregatorFunctionResult:
    mongo_client = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )
    logger.info(f"Aggregator invoked for session {session_id} and round {round_id}")
    try:

        result_dao = ClientResultDao(mongo_client)
        parameter_dao = ParameterDao(mongo_client)
        logger.debug(f"Establishing database connection")
        previous_results: Iterator[ClientResult] = result_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )

        if not previous_results:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )
        aggregator = FedAvgAggregator()
        if online:
            logger.debug(f"Using online aggregation")
            aggregator = StreamFedAvgAggregator()
        else:
            logger.debug(f"Loading results from database...")
            previous_results = (
                list(previous_results)
                if not isinstance(previous_results, list)
                else previous_results
            )
            logger.debug(f"Loading of {len(previous_results)} results finished")
        logger.debug(f"Starting aggregation...")
        new_parameters, test_results = aggregator.aggregate(previous_results)
        logger.debug(f"Aggregation finished")

        global_test_metrics = None
        if test_data:
            logger.debug(f"Evaluating model")
            model_dao = ModelDao(mongo_client)
            # Load model and latest weights
            serialized_model: SerializedModel = model_dao.load(session_id=session_id)
            test_data = DatasetLoaderBuilder.from_config(test_data).load()
            cardinality = test_data.cardinality()
            test_data = test_data.batch(test_batch_size)
            model: tf.keras.Model = tf.keras.models.model_from_json(
                serialized_model.model_json
            )
            model.set_weights(new_parameters)
            if not serialized_model.loss or not serialized_model.optimizer:
                raise AggregationError("If compiled=True, a loss has to be specified")
            model.compile(
                optimizer=tf.keras.optimizers.get(serialized_model.optimizer),
                loss=tf.keras.losses.get(serialized_model.loss),
                metrics=serialized_model.metrics or [],
            )
            evaluation_result = model.evaluate(test_data, return_dict=True)
            global_test_metrics = TestMetrics(
                cardinality=cardinality, metrics=evaluation_result
            )

        logger.debug(f"Serializing model")
        serialized_params_str = WeightsSerializerBuilder.from_config(
            serializer
        ).serialize(new_parameters)

        serialized_params = SerializedParameters(
            blob=serialized_params_str, serializer=serializer
        )

        new_round_id = round_id + 1
        logger.debug(f"Saving model to database")
        parameter_dao.save(
            session_id=session_id, round_id=new_round_id, params=serialized_params
        )
        logger.debug(f"Finished...")

        results_processed = result_dao.count_results_for_round(
            session_id=session_id, round_id=round_id
        )
        if delete_results_after_finish:
            logger.debug(f"Deleting individual results...")
            result_dao.delete_results_for_round(
                session_id=session_id, round_id=round_id
            )

        return AggregatorFunctionResult(
            new_round_id=new_round_id,
            num_clients=results_processed,
            test_results=test_results,
            global_test_results=global_test_metrics,
        )
    except (SerializationError, PersistenceError) as e:
        raise AggregationError(e) from e
    finally:
        mongo_client.close()


class ParameterAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(
        self, client_results: Iterator[ClientResult]
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        pass


class FedAvgAggregator(ParameterAggregator):
    def _aggregate(
        self, parameters: List[List[np.ndarray]], weights: List[float]
    ) -> List[np.ndarray]:
        # Partially from https://github.com/adap/flower/blob/
        # 570788c9a827611230bfa78f624a89a6630555fd/src/py/flwr/server/strategy/aggregate.py#L26
        num_examples_total = sum(weights)
        weighted_weights = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in zip(parameters, weights)
        ]

        # noinspection PydanticTypeChecker,PyTypeChecker
        weights_prime: List[np.ndarray] = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        client_parameters: List[List[np.ndarray]] = []
        client_cardinalities: List[int] = []
        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            del client_result.parameters
            cardinality = client_result.cardinality

            # Check if cardinality is valid and handle accordingly
            if cardinality in [
                tf.data.UNKNOWN_CARDINALITY,
                tf.data.INFINITE_CARDINALITY,
            ]:
                if not default_cardinality:
                    raise UnknownCardinalityError(
                        f"Cardinality for client result invalid. "
                    )
                else:
                    cardinality = default_cardinality

            client_parameters.append(params)
            client_cardinalities.append(cardinality)
            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return (
            self._aggregate(client_parameters, client_cardinalities),
            client_metrics or None,
        )


def chunks(iterator: Iterator, n) -> Iterator[List]:
    buffer = []
    for el in iterator:
        if len(buffer) < n:
            buffer.append(el)
        if len(buffer) == n:
            yield buffer
            buffer = []
    else:
        if len(buffer) > 0:
            yield buffer


class StreamFedAvgAggregator(FedAvgAggregator):
    def __init__(self, chunk_size: int = 25):
        self.chunk_size = chunk_size

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        curr_global_params: Parameters = None
        curr_sum_weights = 0
        client_metrics: List[TestMetrics] = []
        for results_chunk in chunks(client_results, self.chunk_size):
            params_buffer, card_buffer = [], []
            for client_result in results_chunk:
                params = deserialize_parameters(client_result.parameters)
                del client_result.parameters
                cardinality = client_result.cardinality

                # Check if cardinality is valid and handle accordingly
                if cardinality in [
                    tf.data.UNKNOWN_CARDINALITY,
                    tf.data.INFINITE_CARDINALITY,
                ]:
                    if not default_cardinality:
                        raise UnknownCardinalityError(
                            f"Cardinality for client result invalid. "
                        )
                    else:
                        cardinality = default_cardinality

                params_buffer.append(params)
                card_buffer.append(cardinality)
                if client_result.test_metrics:
                    client_metrics.append(client_result.test_metrics)
            if curr_global_params is None:
                curr_global_params = self._aggregate(params_buffer, card_buffer)
            else:
                curr_global_params = self._aggregate(
                    [curr_global_params, *params_buffer],
                    [curr_sum_weights, *card_buffer],
                )
            curr_sum_weights += sum(card_buffer)

        return curr_global_params, client_metrics or None
