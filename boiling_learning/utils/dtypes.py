from __future__ import annotations

from typing import Hashable, List, Mapping, Optional, Tuple, TypeVar, Union

import tensorflow as tf
from bidict import bidict
from tensorflow.types.experimental import TensorLike
from typing_extensions import TypedDict

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

tf_str_dtype_bidict = bidict(
    (dtype.name, dtype)
    for dtype in (
        tf.float16,
        tf.float32,
        tf.float64,
        tf.bfloat16,
        tf.complex64,
        tf.complex128,
        tf.int8,
        tf.int32,
        tf.int64,
        tf.uint8,
        tf.uint16,
        tf.uint32,
        tf.uint64,
        tf.int16,
        tf.bool,
        tf.string,
        tf.qint8,
        tf.qint16,
        tf.qint32,
        tf.quint8,
        tf.quint16,
        tf.resource,
        tf.variant,
    )
)


class EncodedElementSpec(TypedDict):
    dtype: str
    name: str
    shape: Tuple[Optional[int], ...]


def encode_element_spec(
    element_spec: NestedTypeSpec,
) -> NestedStructure[EncodedElementSpec]:
    if tf.nest.is_nested(element_spec):
        return {
            'nested': True,
            'contents': map_values(encode_element_spec, element_spec),
        }
    else:
        return {
            'nested': False,
            'contents': {
                'dtype': tf_str_dtype_bidict.inverse[element_spec.dtype],
                'name': element_spec.name,
                'shape': element_spec.shape,
            },
        }


def decode_element_spec(
    obj: NestedStructure[EncodedElementSpec],
) -> NestedTypeSpec:
    nested, contents = obj['nested'], obj['contents']

    if nested:
        return map_values(decode_element_spec, contents)

    return tf.TensorSpec(
        shape=contents['shape'],
        name=contents['name'],
        dtype=tf_str_dtype_bidict[contents['dtype']],
    )


def auto_spec(elem: NestedTensorLike) -> NestedTypeSpec:
    try:
        return tf.type_spec_from_value(elem)
    except TypeError:
        return map_values(auto_spec, elem)


def auto_dtype(elem: NestedTensorLike) -> NestedStructure[tf.DType]:
    try:
        return tf.constant(elem).dtype
    except (TypeError, AttributeError, ValueError):
        return map_values(auto_dtype, elem)
