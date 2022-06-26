from __future__ import annotations

from typing import Any, Dict, List

import autokeras as ak
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from classes import typeclass

from boiling_learning.io import json


class LayersBlock(ak.engine.block.Block):
    def __init__(self, layers: List[tf.keras.layers.Layer], **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = layers

    def build(  # pylint: disable=signature-differs
        self,
        hp: kt.HyperParameters,
        inputs: List[ak.Node],
    ) -> tf.keras.Model:
        inputs = tf.nest.flatten(inputs)
        input_node = inputs[0]
        output_node = input_node

        for layer in self.layers:
            layer._init_set_name(None)  # pylint: disable=protected-access
            output_node = layer(output_node)

        return output_node

    def get_config(self) -> Dict[str, Any]:
        return {'layers': clean_config([layer.get_config() for layer in self.layers])}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LayersBlock:
        return cls(
            [tf.keras.layers.Layer.from_config(layer_config) for layer_config in config['layers']]
        )


@typeclass
def clean_config(config: Any) -> json.JSONDataType:
    '''Return clean configuration for JSON serialization'''


@clean_config.instance(object)
def _clean_config_default(config: Any) -> json.JSONDataType:
    return config


@clean_config.instance(np.ndarray)
def _clean_config_numpy_arrays(config: np.ndarray) -> json.JSONDataType:
    return float(config)


@clean_config.instance(dict)
def _clean_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: clean_config(value) for key, value in config.items()}


@clean_config.instance(list)
def _clean_config_list(config: List[Any]) -> List[Any]:
    return [clean_config(value) for value in config]
