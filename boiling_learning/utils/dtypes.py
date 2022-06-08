from __future__ import annotations

from typing import Hashable, List, Mapping, Tuple, TypeVar, Union

import tensorflow as tf
from tensorflow.types.experimental import TensorLike

from boiling_learning.utils.functional import map_values

_T = TypeVar('_T')
NestedStructure = Union[
    _T,
    List['NestedStructure[_T]'],
    Tuple['NestedStructure[_T]', ...],
    Mapping[Hashable, 'NestedStructure[_T]'],
]
NestedTypeSpec = NestedStructure[tf.TypeSpec]
NestedTensorLike = NestedStructure[TensorLike]


def auto_spec(elem: NestedTensorLike) -> NestedTypeSpec:
    try:
        return tf.type_spec_from_value(elem)
    except TypeError:
        return map_values(auto_spec, elem)
