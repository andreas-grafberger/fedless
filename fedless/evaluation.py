import pymongo

from fedless.data import DatasetLoaderBuilder, DatasetNotLoadedError
from fedless.models import (
    EvaluatorResult,
    MongodbConnectionConfig,
    ModelLoaderConfig,
    SimpleModelLoaderConfig,
    TestMetrics,
    DatasetLoaderConfig,
)
from fedless.persistence import ModelDao, ParameterDao, PersistenceError
from fedless.serialization import ModelLoaderBuilder, SerializationError


class EvaluationError(BaseException):
    """Something went wrong during evaluation"""


def default_evaluation_handler(
    database: MongodbConnectionConfig,
    session_id: str,
    round_id: int,
    test_data: DatasetLoaderConfig,
    batch_size: int = 10,
) -> EvaluatorResult:
    db = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )
    try:
        # Create daos to access database
        model_dao = ModelDao(db=db)
        parameter_dao = ParameterDao(db=db)

        # Load model and latest weights
        model = model_dao.load(session_id=session_id)
        latest_params = parameter_dao.load(session_id=session_id, round_id=round_id)
        model_config = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_params,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )
        model = ModelLoaderBuilder.from_config(model_config).load()

        # Load data
        test_data = DatasetLoaderBuilder.from_config(test_data).load()
        cardinality = test_data.cardinality()

        test_data = test_data.batch(batch_size)

        evaluation_result = model.evaluate(test_data, return_dict=True)
        test_metrics = TestMetrics(cardinality=cardinality, metrics=evaluation_result)

    except (
        PersistenceError,
        SerializationError,
        DatasetNotLoadedError,
        ValueError,
    ) as e:
        raise EvaluationError(e) from e
    finally:
        db.close()
    return EvaluatorResult(metrics=test_metrics)
