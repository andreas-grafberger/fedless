import logging

from pydantic import ValidationError

from fedless.aggregation import default_aggregation_handler, AggregationError
from fedless.providers import openwhisk_action_handler
from fedless.models import AggregatorFunctionParams

logging.basicConfig(level=logging.DEBUG)


@openwhisk_action_handler((ValidationError, AggregationError))
def main(request):
    config = AggregatorFunctionParams.parse_obj(request["body"])

    return default_aggregation_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        database=config.database,
        serializer=config.serializer,
        online=config.online,
        test_data=config.test_data,
        test_batch_size=config.test_batch_size,
    )
