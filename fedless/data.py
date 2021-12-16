import abc
import json
import logging
import os
import tempfile
from functools import reduce
from json import JSONDecodeError
from pathlib import Path
from typing import Union, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
import tensorflow as tf
from pydantic import validate_arguments, AnyHttpUrl
from requests import RequestException

from fedless.cache import cache
from fedless.models import LEAFConfig, DatasetLoaderConfig, LeafDataset, MNISTConfig

logger = logging.getLogger(__name__)


class DatasetNotLoadedError(Exception):
    """Dataset could not be loaded"""


class DatasetFormatError(DatasetNotLoadedError):
    """Source file containing data is malformed or otherwise invalid"""


def merge_datasets(datasets: Iterator[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Merge the given datasets into one by concatenating them
    :param datasets: Iterator with all datasets
    :return: Final combined dataset
    :raises TypeError in tf.data.Dataset.concatenate
    """
    return reduce(tf.data.Dataset.concatenate, datasets)


class DatasetLoader(abc.ABC):
    """Load arbitrary datasets"""

    @abc.abstractmethod
    def load(self) -> tf.data.Dataset:
        """Load dataset"""
        pass


class LEAF(DatasetLoader):
    """
    Utility class to load and process the LEAF datasets as published in
    https://arxiv.org/pdf/1812.01097.pdf and https://github.com/TalwalkarLab/leaf
    """

    @validate_arguments
    def __init__(
        self,
        dataset: LeafDataset,
        location: Union[AnyHttpUrl, Path],
        http_params: Dict = None,
        user_indices: Optional[List[int]] = None,
    ):
        """
        Create dataset loader for the specified source
        :param dataset: Dataset name, one of :py:class:`fedless.models.LeafDataset`
        :param location: Location of dataset partition in form of a json file.
        :param http_params: Additional parameters to send with http request. Only used when location is an URL
         Use location:// to load from disk. For valid entries see :py:meth:`requests.get`
        """
        self.dataset = dataset
        self.source = location
        self.http_params = http_params
        self.user_indices = user_indices
        self._users = []

        if dataset not in [LeafDataset.FEMNIST, LeafDataset.SHAKESPEARE]:
            raise NotImplementedError()

    def _iter_dataset_files(self) -> Iterator[Union[AnyHttpUrl, Path]]:
        if isinstance(self.source, AnyHttpUrl):
            yield self.source
        elif isinstance(self.source, Path) and self.source.is_dir():
            for file in self.source.iterdir():
                if file.is_file() and file.suffix == ".json":
                    yield file
        else:
            yield self.source

    @property
    def users(self):
        return self._users

    def _convert_dict_to_dataset(
        self, file_content: Dict, user_indices: List[int] = None
    ) -> tf.data.Dataset:
        try:
            users = file_content["users"]
            user_data = file_content["user_data"]
            self._users = users
            for i, user in enumerate(users):
                if not user_indices or i in user_indices:
                    yield tf.data.Dataset.from_tensor_slices(
                        self._process_user_data(user_data[user])
                    )
        except (KeyError, TypeError, ValueError) as e:
            raise DatasetFormatError(e) from e

    def _process_user_data(self, user_data: Dict) -> Tuple:
        if self.dataset == LeafDataset.SHAKESPEARE:
            vocabulary = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
            vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                standardize=None,
                split=tf.strings.bytes_split,
                vocabulary=[c for c in vocabulary],
            )
            return vectorizer(tf.convert_to_tensor(user_data["x"])), vectorizer(
                tf.convert_to_tensor(user_data["y"])
            )

        return user_data["x"], user_data["y"]

    def _process_all_sources(self) -> Iterator[tf.data.Dataset]:
        for source in self._iter_dataset_files():
            file_content: Dict = self._read_source(source)
            for dataset in self._convert_dict_to_dataset(
                file_content, user_indices=self.user_indices
            ):
                yield dataset

    def _read_source(self, source: Union[AnyHttpUrl, Path]) -> Dict:
        if isinstance(source, AnyHttpUrl):
            return self._fetch_url(source)
        else:
            return self._read_file_content(source)

    def _fetch_url(self, url: str):
        try:
            response = requests.get(url, params=self.http_params)
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            raise DatasetFormatError(f"Invalid JSON returned from ${url}") from e
        except RequestException as e:
            raise DatasetNotLoadedError(e) from e

    @classmethod
    def _read_file_content(cls, path: Path) -> Dict:
        try:
            with path.open() as f:
                return json.load(f)
        except (JSONDecodeError, ValueError) as e:
            raise DatasetFormatError(e) from e
        except (IOError, OSError) as e:
            raise DatasetNotLoadedError(e) from e

    @cache
    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        sources = self._process_all_sources()
        try:
            return merge_datasets(sources)
        except TypeError as e:
            raise DatasetFormatError(e) from e


class MNIST(DatasetLoader):
    def __init__(
        self,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}

    @cache
    def load(self) -> tf.data.Dataset:
        response = requests.get(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            proxies=self.proxies,
        )
        fp, path = tempfile.mkstemp()
        with os.fdopen(fp, "wb") as f:
            f.write(response.content)

        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        if self.split.lower() == "train":
            features, labels = x_train, y_train
        elif self.split.lower() == "test":
            features, labels = x_test, y_test
        else:
            raise DatasetNotLoadedError(f"Mnist split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        def _scale_features(features, label):
            return tf.cast(features, tf.float32) / 255.0, tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        return ds.map(_scale_features)


class DatasetLoaderBuilder:
    """Convenience class to construct loaders from config"""

    @staticmethod
    def from_config(config: DatasetLoaderConfig) -> DatasetLoader:
        """
        Construct loader from config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "leaf":
            params: LEAFConfig = config.params
            return LEAF(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        elif config.type == "mnist":
            params: MNISTConfig = config.params
            return MNIST(
                split=params.split, indices=params.indices, proxies=params.proxies
            )
        else:
            raise NotImplementedError(
                f"Dataset loader {config.type} is not implemented"
            )
