from contextlib import AbstractContextManager
from typing import Any, Union, Callable, Iterator

import bson
import pymongo
from gridfs import GridFS
from gridfs.errors import GridFSError
from pymongo.collection import Collection
from pymongo.errors import InvalidName, PyMongoError, ConnectionFailure, BSONError

from fedless.models import (
    ClientResult,
    MongodbConnectionConfig,
    ClientConfig,
    SerializedParameters,
    SerializedModel,
)


class PersistenceError(Exception):
    """Base exception for persistence errors"""


class PersistenceValueError(PermissionError):
    """Unexpected or invalid value encountered"""


class StorageConnectionError(PersistenceError):
    """Connection to storage resource (e.g. database) could not be established"""


class DocumentNotStoredException(PersistenceError):
    """Client result could not be stored"""


class DocumentAlreadyExistsException(PersistenceError):
    """A result for this client already exists"""


class DocumentNotLoadedException(PersistenceError):
    """Client result could not be loaded"""


def wrap_pymongo_errors(func: Callable) -> Callable:
    """Decorator to wrap all unhandled pymongo exceptions as persistence errors"""

    # noinspection PyMissingOrEmptyDocstring
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (PyMongoError, BSONError) as e:
            raise PersistenceError(e) from e

    return wrapped_function


class MongoDbDao(AbstractContextManager):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str,
        database: str = "fedless",
    ):
        """
        Connect to mongodb database
        :param db: mongodb url or config object of type :class:`MongodbConnectionConfig`
        :param database: database name
        :param collection: collection name
        """
        self.db = db
        self.database = database
        self.collection = collection

        if isinstance(db, str):
            self._client = pymongo.MongoClient(db)
        elif isinstance(db, MongodbConnectionConfig):
            self._client = pymongo.MongoClient(
                host=db.host,
                port=db.port,
                username=db.username,
                password=db.password,
            )
        else:
            self._client = db

        try:
            self._collection: Collection = self._client[database][collection]
        except InvalidName as e:
            raise StorageConnectionError(e) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._client.close()


class ClientResultDao(MongoDbDao):
    """Store client results in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "results",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        round_id: int,
        client_id: str,
        result: Union[dict, ClientResult],
        overwrite: bool = True,
    ) -> Any:
        if isinstance(result, ClientResult):
            result = result.dict()

        if (
            not overwrite
            and self._collection.find_one(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                }
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client result for session {session_id} and round {round_id} for client {client_id} already exists. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(result))
            self._collection.replace_one(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                },
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                    "file_id": file_id,
                },
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def load(
        self,
        session_id: str,
        round_id: int,
        client_id: str,
    ) -> ClientResult:
        try:
            obj_dict = self._collection.find_one(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                },
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Client result for session {session_id} and round {round_id} for client {client_id} not found."
            )

        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Client result for session {session_id} and client {client_id}"
                f"for round {round_id} malformed."
            )
        results_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not results_file:
            raise DocumentNotLoadedException(
                f"GridFS file with results for session {session_id} and client {client_id} "
                f"and round {round_id} not found."
            )
        try:
            return ClientResult.parse_obj(bson.decode(results_file.read()))
        finally:
            results_file.close()

    @wrap_pymongo_errors
    def load_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ) -> Iterator[ClientResult]:
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for result_dict in result_dicts:
            if not result_dict:
                raise DocumentNotLoadedException(
                    f"Client results for session {session_id} and round {round_id} not found."
                )
            if "file_id" not in result_dict:
                raise PersistenceValueError(
                    f"Client result in session {session_id} for round {round_id} malformed."
                )
            results_file = self._gridfs.find_one({"_id": result_dict["file_id"]})
            if not results_file:
                raise DocumentNotLoadedException(
                    f"GridFS file with results in session {session_id} "
                    f"and round {round_id} not found."
                )
            try:
                yield ClientResult.parse_obj(bson.decode(results_file.read()))
            finally:
                results_file.close()

    @wrap_pymongo_errors
    def delete_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def count_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ) -> int:
        try:
            return self._collection.count_documents(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                },
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e


class ClientConfigDao(MongoDbDao):
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "clients",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(self, client: ClientConfig, overwrite: bool = True) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"client_id": client.client_id}) is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"client_id": client.client_id}, client.dict(), upsert=True
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, client_id: str) -> ClientConfig:
        try:
            obj_dict = self._collection.find_one(filter={"client_id": client_id})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Client with id {client_id} not found")
        return ClientConfig.parse_obj(obj_dict)

    @wrap_pymongo_errors
    def load_all(self, session_id: str) -> Iterator[ClientConfig]:
        try:
            obj_dicts = iter(self._collection.find(filter={"session_id": session_id}))
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for client_dict in obj_dicts:
            yield ClientConfig.parse_obj(client_dict)


class ParameterDao(MongoDbDao):
    """Store global model parameters in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "parameters",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        round_id: int,
        params: SerializedParameters,
        overwrite: bool = True,
    ) -> Any:

        if (
            not overwrite
            and self._collection.find_one(
                {"session_id": session_id, "round_id": round_id}
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Parameters for session {session_id} and round {round_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(params.dict()), encoding="utf-8")
            self._collection.replace_one(
                {"session_id": session_id, "round_id": round_id},
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "file_id": file_id,
                },
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(
        self,
        session_id: str,
        round_id: int,
    ) -> SerializedParameters:
        try:
            obj_dict = self._collection.find_one(
                filter={"session_id": session_id, "round_id": round_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} and round {round_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} "
                f"and round {round_id} malformed."
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} and round {round_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def load_latest(self, session_id: str) -> SerializedParameters:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} " f"Expected key file_id"
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def get_latest_round(self, session_id: str) -> int:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if obj_dict is None or "round_id" not in obj_dict:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found or malformed"
            )
        return int(obj_dict["round_id"])


class ModelDao(MongoDbDao):
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "models",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(
        self, session_id: str, model: SerializedModel, overwrite: bool = True
    ) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"session_id": session_id}) is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Model architecture for session {session_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"session_id": session_id},
                {"session_id": session_id, "model": model.dict()},
                upsert=True,
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, session_id: str) -> SerializedModel:
        try:
            obj_dict = self._collection.find_one(filter={"session_id": session_id})
            obj_dict = (
                obj_dict["model"]
                if obj_dict is not None and "model" in obj_dict
                else None
            )

            if obj_dict is None:
                raise DocumentNotLoadedException(f"Client with id {id} not found")

            return SerializedModel.parse_obj(obj_dict)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except KeyError:
            raise PersistenceValueError(
                f"Loaded model architecture for session {session_id} malformed. Expected key parameters"
            )
