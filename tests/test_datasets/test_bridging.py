import numpy as np
import tensorflow as tf

from boiling_learning.datasets.bridging import auto_spec
from boiling_learning.datasets.sliceable import SliceableDataset


def test_auto_spec() -> None:
    sds1 = SliceableDataset.from_sequence('abcd')
    assert auto_spec(sds1[0]) == tf.TensorSpec(shape=(), dtype=tf.string)

    sds2 = SliceableDataset.from_sequence([0, 1, 2])
    assert auto_spec(sds2[0]) == tf.TensorSpec(shape=(), dtype=tf.int32)

    sds3 = SliceableDataset.from_sequence([(0, 'a'), (1, 'b'), (2, 'c')])
    assert auto_spec(sds3[0]) == (
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    sds4 = SliceableDataset.from_sequence([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert auto_spec(sds4[0]) == (tf.TensorSpec(shape=(3), dtype=tf.int32))

    sds5 = SliceableDataset.from_sequence(
        [
            (
                np.random.rand(3, 4),
                {
                    'key1': [np.random.rand(2, 5), np.random.rand(2, 5), np.random.rand(2, 5)],
                    'key2': 'value1',
                    'key3': 10.5,
                },
                False,
            ),
            (
                np.random.rand(3, 4),
                {
                    'key1': [np.random.rand(2, 5), np.random.rand(2, 5)],
                    'key2': 'value2',
                    'key3': 3.14,
                },
                True,
            ),
        ]
    )
    assert auto_spec(sds5[0]) == (
        tf.TensorSpec(shape=(3, 4), dtype=tf.float64),
        {
            'key1': tf.TensorSpec(shape=(3, 2, 5), dtype=tf.float64),
            'key2': tf.TensorSpec(shape=(), dtype=tf.string),
            'key3': tf.TensorSpec(shape=(), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.bool),
    )
