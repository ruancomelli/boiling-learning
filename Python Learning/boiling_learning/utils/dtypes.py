import bidict
import pandas as pd
import tensorflow as tf


tf_str_dtype_bidict = bidict.bidict(
    (dtype.name, dtype)
    for dtype in (
        tf.float16, tf.float32, tf.float64,
        tf.bfloat16,
        tf.complex64, tf.complex128,
        tf.int8, tf.int32, tf.int64,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16,
        tf.bool,
        tf.string,
        tf.qint8, tf.qint16, tf.qint32, tf.quint8,
        tf.quint16,
        tf.resource,
        tf.variant
    )
)


pd_tf_dtype_bidict = bidict.bidict({
    pd.int64: tf.int64,
    pd.float64: tf.float64,
    pd.bool: tf.bool,
    pd.category: tf.string
})
