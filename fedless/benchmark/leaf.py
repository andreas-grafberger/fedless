from typing import Iterable, Optional, Dict, List, Iterator

from tensorflow import keras

from fedless.data import LEAF
from fedless.models import LeafDataset, LEAFConfig


def create_femnist_cnn(
    num_classes: int = 62, small: bool = False
) -> keras.models.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.InputLayer((28 * 28,)))
    model.add(keras.layers.Reshape((28, 28, 1)))
    model.add(
        keras.layers.Convolution2D(
            filters=(16 if small else 32),
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(
        keras.layers.Convolution2D(
            filters=(32 if small else 64),
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense((512 if small else 2048), activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_shakespeare_lstm(
    units: int = 256,
    vocab_size: int = 82,
    sequence_length: int = 80,
    embedding_size: int = 8,
):
    model = keras.Sequential()
    keras.layers.InputLayer((sequence_length, vocab_size))
    model.add(
        keras.layers.Embedding(
            vocab_size,
            embedding_size,
        )
    )
    model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.LSTM(units))
    model.add(keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def split_source_by_users(config: LEAFConfig) -> Iterable[LEAFConfig]:
    loader = LEAF(
        dataset=config.dataset,
        location=config.location,
        http_params=config.http_params,
        user_indices=config.user_indices,
    )
    loader.load()

    for i, _ in enumerate(loader.users):
        if not config.user_indices or i in config.user_indices:
            yield LEAFConfig(
                dataset=config.dataset,
                location=config.location,
                http_params=config.http_params,
                user_indices=[i],
            )


def split_sources_by_users(source_urls: List[LEAFConfig]) -> Iterator[LEAFConfig]:
    for source in source_urls:
        for config in split_source_by_users(source):
            yield config
