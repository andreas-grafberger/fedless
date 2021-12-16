import logging

import azure.functions
from pydantic import ValidationError

from fedless.providers import azure_handler
from fedless.models import InvokerParams
from fedless.client import fedless_mongodb_handler, ClientError

logging.basicConfig(level=logging.DEBUG)


@azure_handler(caught_exceptions=(ValidationError, ValueError, ClientError))
def main(req: azure.functions.HttpRequest):
    config = InvokerParams.parse_obj(req.get_json())

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
    )
