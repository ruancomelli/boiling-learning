from pathlib import Path
from typing import Iterable, List, Tuple

import tensorflow as tf

from boiling_learning.utils import VerboseType


def check_all_paths_exist(
    named_paths: Iterable[Tuple[str, Path]], *, verbose: VerboseType = False
) -> None:
    for name, path in named_paths:
        if not path.exists():
            raise RuntimeError(f'path to "{name}" does not exist: {path}')

        if verbose:
            print(f'{name}: {path}')


def initialize_gpus() -> tf.distribute.Strategy:
    # See <https://www.tensorflow.org/xla/tutorials/autoclustering_xla>
    tf.config.optimizer.set_jit(True)  # Enable XLA.

    cpus: List[str] = tf.config.list_physical_devices('CPU')
    gpus: List[str] = tf.config.list_physical_devices('GPU')

    print('Available CPUs:', cpus)
    print('Available GPUs:', gpus)

    if not gpus:
        raise RuntimeError('No GPUs connected.')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return tf.distribute.MirroredStrategy(
        # trying to reduce memory usage
        cross_device_ops=tf.distribute.NcclAllReduce(num_packs=0)
    )
