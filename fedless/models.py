from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, List
from urllib import parse

import numpy as np
from pydantic import (
    Field,
    BaseModel,
    AnyHttpUrl,
    validator,
    BaseSettings,
    PositiveInt,
    StrictBytes,
)
from pydantic.fields import ModelField


def params_validate_types_match(
    params: BaseModel, values: Dict, field: Optional[ModelField] = None
):
    """
    Custom pydantic validator used together with :func:`pydantic.validator`.
    Can be used for :class:`BaseModel`'s that contain both a type and params attribute.
    It checks if the parameter set contains a type attribute and if it matches the type
    specified in the model itself.

    :param params: Model of parameter set
    :param values: Dictionary with previously checked attributes. Has to contain key "type"
    :params field: Supplied by pydantic, set to None for easier testability
    :return: params if they are valid
    :raises ValueError, TypeError
    """
    # Do not throw error but accept empty params. This allows one to
    # not specify params if the type allows it and e.g. just uses default values
    if field and not field.required and params is None:
        return params

    try:
        expected_type = values["type"]
        params_type = getattr(params, "type")
    except KeyError:
        raise ValueError(f'Required field "type" not given.')
    except AttributeError:
        raise ValueError(
            f'Field "type" is missing in the class definition of model {params.__class__}'
        )

    if expected_type != params_type:
        raise TypeError(
            f"Given values for parameters of type {params_type} do not match the expected type {expected_type}"
        )
    return params


Parameters = List[np.ndarray]


class SerializedModel(BaseModel):
    model_json: str
    optimizer: Union[str, Dict]
    loss: Union[str, Dict]
    metrics: List[str]


class TestMetrics(BaseModel):
    cardinality: int = Field(
        description="tf.data.INFINITE_CARDINALITY if the dataset contains an infinite number of elements or "
        "tf.data.UNKNOWN_CARDINALITY if the analysis fails to determine the number of elements in the dataset "
        "Source: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality"
    )
    metrics: Dict = Field(description="Dictionary mapping from metric name to value")


class H5FullModelSerializerConfig(BaseModel):
    """Configuration parameters for this serializer"""

    type: str = Field("h5", const=True)
    save_traces: bool = True


class NpzWeightsSerializerConfig(BaseModel):
    """Configuration parameters for this serializer"""

    type: str = Field("npz", const=True)
    compressed: bool = False


class BinaryStringFormat(str, Enum):
    BASE64 = "base64"
    NONE = "none"


class LeafDataset(str, Enum):
    """
    Officially supported datasets
    """

    FEMNIST = "femnist"
    REDDIT = "reddit"
    CELEBA = "celeba"
    SHAKESPEARE = "shakespeare"
    SENT140 = "sent140"


class LocalDifferentialPrivacyParams(BaseModel):
    l2_norm_clip: float
    noise_multiplier: float
    num_microbatches: Optional[int]


class Hyperparams(BaseModel):
    """Parameters for training and some data processing"""

    batch_size: PositiveInt
    epochs: PositiveInt
    shuffle_data: bool = True
    optimizer: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Optimizer, either string with name of optimizer or "
        "a config dictionary retrieved via tf.keras.optimizers.serialize. ",
    )
    loss: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Name of loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses, or "
        "a config dictionary retrieved via tf.keras.losses.serialize. ",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of metrics to be evaluated by the model",
    )
    local_privacy: Optional[LocalDifferentialPrivacyParams]


class LEAFConfig(BaseModel):
    """Configuration parameters for LEAF dataset loader"""

    type: str = Field("leaf", const=True)
    dataset: LeafDataset
    location: Union[AnyHttpUrl, Path]
    http_params: Dict = None
    user_indices: Optional[List[int]] = None


class MNISTConfig(BaseModel):
    """Configuration parameters for sharded MNIST dataset"""

    type: str = Field("mnist", const=True)
    indices: List[int] = None
    split: str = "train"
    proxies: Optional[Dict] = None


class DatasetLoaderConfig(BaseModel):
    """Configuration for arbitrary dataset loaders"""

    type: str
    params: Union[LEAFConfig, MNISTConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class OpenwhiskActionConfig(BaseModel):
    """Info to describe different functions deployed in an openwhisk cluster"""

    type: str = Field("openwhisk", const=True)

    namespace: str = "guest"
    package: str = "default"
    name: str
    auth_token: str
    api_host: str
    self_signed_cert: bool


class OpenwhiskWebActionConfig(BaseModel):
    type: str = Field("openwhisk-web", const=True)
    self_signed_cert: bool = True
    endpoint: str
    token: Optional[str]


class ApiGatewayLambdaFunctionConfig(BaseModel):
    """Lambda function deployed via Api Gateway. All requests time out after 30 seconds due to fixed limit"""

    type: str = Field("lambda", const=True)
    apigateway: str
    api_key: Optional[str]


class GCloudFunctionConfig(BaseModel):
    """Google cloud function"""

    type: str = Field("gcloud", const=True)
    url: str


class OpenFaasFunctionConfig(BaseModel):
    """OpenFaas function"""

    type: str = Field("openfaas", const=True)
    url: str


class AzureFunctionHTTPConfig(BaseModel):
    """Azure function"""

    type: str = Field("azure", const=True)
    trigger_url: str


class FunctionInvocationConfig(BaseModel):
    """Necessary information to invoke a function"""

    type: str
    params: Union[
        OpenwhiskActionConfig,
        ApiGatewayLambdaFunctionConfig,
        GCloudFunctionConfig,
        OpenwhiskWebActionConfig,
        AzureFunctionHTTPConfig,
        OpenFaasFunctionConfig,
    ]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class InvocationResult(BaseModel):
    """Returned by invoker functions"""

    session_id: str
    round_id: int
    client_id: str
    test_metrics: Optional[TestMetrics] = None


class MongodbConnectionConfig(BaseSettings):
    """
    Data class holding connection info for a MongoDB instance
    Automatically tries to fill in missing values from environment variables
    """

    host: str = Field(...)
    port: int = Field(...)
    username: str = Field(...)
    password: str = Field(...)

    @property
    def url(self) -> str:
        """Return url representation"""
        return f"mongodb://{parse.quote(self.username)}:{parse.quote(self.password)}@{self.host}:{self.port}"

    class Config:
        env_prefix = "fedless_mongodb_"


class ModelSerializerConfig(BaseModel):
    """Configuration object for arbitrary model serializers of type :class:`ModelSerializer`"""

    type: str
    params: Optional[Union[H5FullModelSerializerConfig]]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class WeightsSerializerConfig(BaseModel):
    """Configuration for parameters serializers of type :class:`WeightsSerializer`"""

    type: str
    params: Union[NpzWeightsSerializerConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class SerializedParameters(BaseModel):
    """Parameters as serialized blob with information on how to deserialize it"""

    blob: Union[StrictBytes, str]
    serializer: WeightsSerializerConfig
    string_format: BinaryStringFormat = BinaryStringFormat.NONE


class LocalPrivacyGuarantees(BaseModel):
    eps: float
    delta: float
    rdp: Optional[List]
    orders: Optional[List]
    steps: Optional[int]


class ClientResult(BaseModel):
    """Result of client function execution"""

    parameters: SerializedParameters
    history: Optional[Dict]
    test_metrics: Optional[TestMetrics]
    cardinality: int = Field(
        description="tf.data.INFINITE_CARDINALITY if the dataset contains an infinite number of elements or "
        "tf.data.UNKNOWN_CARDINALITY if the analysis fails to determine the number of elements in the dataset "
        "(e.g. when the dataset source is a file). "
        "Source: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality"
    )
    privacy_guarantees: Optional[LocalPrivacyGuarantees]


class ClientResultStorageObject(BaseModel):
    """Client Result persisted in database with corresponding client identifier"""

    key: str
    result: ClientResult


class PayloadModelLoaderConfig(BaseModel):
    """Configuration parameters required for :class:`PayloadModelLoader`"""

    type: str = Field("payload", const=True)
    payload: Union[StrictBytes, str]
    serializer: ModelSerializerConfig = ModelSerializerConfig(type="h5")


class SimpleModelLoaderConfig(BaseModel):
    """Configuration parameters required for :class:`SimpleModelLoader`"""

    type: str = Field("simple", const=True)

    params: SerializedParameters
    model: str = Field(
        description="Json representation of model architecture. "
        "Created via tf.keras.Model.to_json()"
    )
    compiled: bool = False
    optimizer: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Optimizer, either string with name of optimizer or "
        "a config dictionary retrieved via tf.keras.optimizers.serialize.",
    )
    loss: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Loss, either string with name of loss or "
        "a config dictionary retrieved via tf.keras.losses.serialize.",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of metrics to be evaluated by the model",
    )


class ModelLoaderConfig(BaseModel):
    """Configuration for arbitrary :class:`ModelLoader`'s"""

    type: str
    params: Union[PayloadModelLoaderConfig, SimpleModelLoaderConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class ClientConfig(BaseModel):
    client_id: str
    session_id: str
    function: FunctionInvocationConfig
    data: DatasetLoaderConfig
    hyperparams: Hyperparams
    test_data: Optional[DatasetLoaderConfig]
    compress_model: bool = False


class InvokerParams(BaseModel):
    """Parameters to run invoker function similarly as proposed by FedKeeper"""

    session_id: str
    round_id: int
    client_id: str
    database: MongodbConnectionConfig
    evaluate_only: bool = False
    http_headers: Optional[Dict] = None
    http_proxies: Optional[Dict] = None


class ClientInvocationParams(BaseModel):
    """Convenience class to directly parse and serialize loaders and hyperparameters"""

    data: DatasetLoaderConfig
    model: ModelLoaderConfig
    hyperparams: Hyperparams
    test_data: Optional[DatasetLoaderConfig]


class AggregatorFunctionParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    serializer: WeightsSerializerConfig = WeightsSerializerConfig(
        type="npz", params=NpzWeightsSerializerConfig(compressed=False)
    )
    online: bool = False
    test_data: Optional[DatasetLoaderConfig]
    test_batch_size: int = 512


class AggregatorFunctionResult(BaseModel):
    new_round_id: int
    num_clients: int
    test_results: Optional[List[TestMetrics]]
    global_test_results: Optional[TestMetrics]


class EvaluatorParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    test_data: DatasetLoaderConfig
    batch_size: int = 128
    metrics: List[str] = ["accuracy"]


class EvaluatorResult(BaseModel):
    metrics: TestMetrics


class OpenwhiskClusterConfig(BaseModel):
    type: str = Field("openwhisk", const=True)
    apihost: str
    auth: str
    insecure: bool = True
    namespace: str = "guest"
    package: str = "default"


class GCloudProjectConfig(BaseModel):
    type: str = Field("gcloud", const=True)
    account: str
    project: str


class FaaSProviderConfig(BaseModel):
    type: str
    params: Union[OpenwhiskClusterConfig, GCloudProjectConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class OpenwhiskFunctionDeploymentConfig(BaseModel):
    type: str = Field("openwhisk", const=True)
    name: str
    main: Optional[str]
    file: str
    image: str
    memory: int
    timeout: int
    web: str = "raw"
    web_secure: bool = False


class GCloudFunctionDeploymentConfig(BaseModel):
    type: str = Field("gcloud", const=True)
    name: str
    directory: str
    memory: int
    timeout: int
    wheel_url: str
    entry_point: Optional[str] = None
    runtime: str = "python38"
    max_instances: int = 100
    trigger_http: bool = True
    allow_unauthenticated: bool = True


class FunctionDeploymentConfig(BaseModel):
    type: str
    params: Union[OpenwhiskFunctionDeploymentConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )
