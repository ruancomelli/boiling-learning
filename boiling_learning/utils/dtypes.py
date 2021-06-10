from typing import Mapping, Sequence

import bidict
import funcy
import tensorflow as tf

from boiling_learning.utils.functional import map_values

tf_str_dtype_bidict = bidict.bidict(
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


def encode_element_spec(element_spec):
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


def decode_element_spec(obj):
    nested, contents = obj['nested'], obj['contents']
    if nested:
        return map_values(decode_element_spec, contents)
    else:
        return tf.TensorSpec(
            shape=contents['shape'],
            name=contents['name'],
            dtype=tf_str_dtype_bidict[contents['dtype']],
        )


def auto_spec(elem):
    if hasattr(elem, 'dtype'):
        return elem.dtype
    else:
        if isinstance(elem, Sequence):
            return funcy.walk(auto_spec, elem)
        elif isinstance(elem, Mapping):
            return funcy.walk_values(auto_spec, elem)
        else:
            return elem.dtype


def new_py_function(func, inp, Tout, name=None):
    # Source: <https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000>

    def _tensor_spec_to_dtype(v):
        return v.dtype if isinstance(v, tf.TensorSpec) else v

    def _dtype_to_tensor_spec(v):
        return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(
            inp, flat_inp, expand_composites=True
        )
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name,
    )
    spec_out = tf.nest.map_structure(
        _dtype_to_tensor_spec, Tout, expand_composites=True
    )
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out
