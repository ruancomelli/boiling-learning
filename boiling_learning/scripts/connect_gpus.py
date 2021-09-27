from typing import Optional

import tensorflow as tf

from boiling_learning.utils import print_verbose


def main(
    require_gpu: bool = True,
    nvidia_output: Optional[str] = None,
    verbose: bool = True,
) -> tf.distribute.Strategy:
    # See <https://www.tensorflow.org/xla/tutorials/autoclustering_xla>
    tf.config.optimizer.set_jit(True)  # Enable XLA.

    cpus = tf.config.list_physical_devices('CPU')
    gpus = tf.config.list_physical_devices('GPU')

    print_verbose(verbose, 'Available CPUs:', cpus)
    print_verbose(verbose, 'Available GPUs:', gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        print_verbose(verbose, 'Connected to GPUs.')
        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce(num_packs=0)
        )
    elif nvidia_output is not None and nvidia_output.startswith('GPU'):
        raise RuntimeError(
            'command *nvidia-smi -L* found GPUs, but TensorFlow could not connect to them.'
        )
    elif require_gpu:
        raise RuntimeError('No GPUs connected.')
    else:
        print_verbose(verbose, 'No GPUs connected.')
        return tf.distribute.get_strategy()
