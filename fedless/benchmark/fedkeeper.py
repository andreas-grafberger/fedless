from typing import Iterator, Optional, Dict

import numpy as np
from tensorflow import keras

from fedless.models import (
    DatasetLoaderConfig,
    MNISTConfig,
)


# Helper functions to create dataset shards / model


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int, proxies: Optional[Dict] = None
) -> Iterator[DatasetLoaderConfig]:
    if n_shards % n_devices != 0:
        raise ValueError(
            f"Can not equally distribute {n_shards} dataset shards among {n_devices} devices..."
        )

    (_, y_train), (_, _) = keras.datasets.mnist.load_data()
    num_train_examples, *_ = y_train.shape

    sorted_labels_idx = np.argsort(y_train, kind="stable")
    sorted_labels_idx_shards = np.split(sorted_labels_idx, n_shards)
    shards_per_device = len(sorted_labels_idx_shards) // n_devices
    np.random.shuffle(sorted_labels_idx_shards)

    for client_idx in range(n_devices):
        client_shards = sorted_labels_idx_shards[
            client_idx * shards_per_device : (client_idx + 1) * shards_per_device
        ]
        indices = np.concatenate(client_shards)
        # noinspection PydanticTypeChecker,PyTypeChecker
        yield DatasetLoaderConfig(
            type="mnist", params=MNISTConfig(indices=indices.tolist(), proxies=proxies)
        )


def create_mnist_cnn(num_classes=10):
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer((28, 28)),
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(
                32,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                64,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
