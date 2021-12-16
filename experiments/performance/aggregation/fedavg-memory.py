from typing import Iterator

import click

from fedless.aggregation import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)
from fedless.benchmark.leaf import create_femnist_cnn
from fedless.models import (
    ClientResult,
    SerializedParameters,
    NpzWeightsSerializerConfig,
    WeightsSerializerConfig,
)
from fedless.serialization import NpzWeightsSerializer


@click.command()
@click.option("--stream/--no-stream", default=False)
@click.option("--num-models", default=200)
@click.option("--large-models/--small-models", default=False)
@click.option("--chunk-size", default=25)
def run(stream: bool, num_models: int, large_models: bool, chunk_size: int):
    def create_models() -> Iterator[ClientResult]:
        for i in range(num_models):
            params = create_femnist_cnn(small=not large_models).get_weights()
            weights_bytes = NpzWeightsSerializer().serialize(params)
            yield ClientResult(
                parameters=SerializedParameters(
                    blob=weights_bytes,
                    serializer=WeightsSerializerConfig(
                        type="npz", params=NpzWeightsSerializerConfig()
                    ),
                ),
                cardinality=i + 1,
            )

    aggregator = (
        StreamFedAvgAggregator(chunk_size=chunk_size) if stream else FedAvgAggregator()
    )

    aggregator.aggregate(create_models())
    print(f"Got final parameters!")


if __name__ == "__main__":
    run()
