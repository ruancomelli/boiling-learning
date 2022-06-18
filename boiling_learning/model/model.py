from __future__ import annotations

import enum
import json as _json
import typing
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
from pint import Quantity

from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.model.layers import ImageNormalization, RandomBrightness
from boiling_learning.utils.units import unit_registry as ureg
from boiling_learning.utils.utils import resolve

_CUSTOM_LAYERS = (ImageNormalization, RandomBrightness)


class ModelArchitecture:
    def __init__(self, model: tf.keras.models.Model) -> None:
        self.model = model

    @classmethod
    def from_inputs_and_outputs(
        cls, inputs: tf.keras.layers.Layer, outputs: tf.keras.layers.Layer
    ) -> ModelArchitecture:
        return cls(tf.keras.models.Model(inputs=inputs, outputs=outputs))

    def __json_encode__(self) -> Dict[str, Any]:
        return _anonymize_model(_json.loads(self.model.to_json()))

    def __describe__(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], json.encode(self))


@serialize.instance(ModelArchitecture)
def _serialize_model(instance: ModelArchitecture, path: Path) -> None:
    path = resolve(path, dir=True)
    model_json_path = path / 'model.json'
    model_weights_path = path / 'weights.h5'

    instance.model.save_weights(str(model_weights_path))
    model_json_path.write_text(instance.model.to_json())


@deserialize.dispatch(ModelArchitecture)
def _deserialize_model(path: Path, _metadata: Metadata) -> ModelArchitecture:
    path = resolve(path)
    model_json_path = path / 'model.json'
    model_weights_path = path / 'weights.h5'

    if not model_json_path.is_file():
        raise FileNotFoundError(str(model_json_path))

    if not model_weights_path.is_file():
        raise FileNotFoundError(str(model_weights_path))

    architecture = ModelArchitecture(
        tf.keras.models.model_from_json(
            model_json_path.read_text(),
            custom_objects={layer.__name__: layer for layer in _CUSTOM_LAYERS},
        )
    )
    architecture.model.load_weights(str(model_weights_path))
    return architecture


def model_memory_usage_in_bytes(
    architecture: ModelArchitecture, *, batch_size: int
) -> Quantity[int]:
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Source: https://stackoverflow.com/a/64359137/5811400

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    model = architecture.model

    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0 * ureg.byte
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += model_memory_usage_in_bytes(layer, batch_size=batch_size)

        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    model_params_count = count_model_parameters(architecture)
    return (
        batch_size * shapes_mem_count + model_params_count
    ) * ureg.byte + internal_model_mem_count


def count_model_parameters(
    architecture: ModelArchitecture, *, trainable: bool = True, non_trainable: bool = True
) -> int:
    model = architecture.model

    if trainable and non_trainable:
        return typing.cast(int, model.count_params())
    elif trainable:
        return sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
    elif non_trainable:
        return sum(tf.keras.backend.count_params(p) for p in model.non_trainable_weights)
    else:
        raise ValueError('at least one of `trainable` and `non_trainable` must be true')


class ProblemType(enum.Enum):
    CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()


def _anonymize_model(model_json: Dict[str, Any]) -> Dict[str, Any]:
    model_config = model_json['config']

    # remove model name
    del model_config['name']

    layer_indices = {layer['name']: index for index, layer in enumerate(model_config['layers'])}

    json_str = _json.dumps(model_json)

    for name, index in layer_indices.items():
        json_str = json_str.replace(name, f'layer_{index}')

    return _json.loads(json_str)
