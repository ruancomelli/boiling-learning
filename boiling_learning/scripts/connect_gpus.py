from typing import Optional

import tensorflow as tf


def main(
    require_gpu: bool = True, nvidia_output: Optional[str] = None
) -> tf.distribute.Strategy:
    # See <https://www.tensorflow.org/xla/tutorials/autoclustering_xla>
    tf.config.optimizer.set_jit(True)  # Enable XLA.

    cpus = tf.config.list_physical_devices('CPU')
    gpus = tf.config.list_physical_devices('GPU')

    print('Available CPUs:', cpus)
    print('Available GPUs:', gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        print('Connected to GPUs.')
        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce(
                num_packs=0
            )  # trying to reduce memory usage
        )
    elif nvidia_output.startswith('GPU'):
        raise RuntimeError(
            'command *nvidia-smi -L* found GPUs, but TensorFlow could not connect to them.'
        )
    elif require_gpu:
        raise RuntimeError('No GPUs connected.')
    else:
        print('No GPUs connected.')
        return tf.distribute.get_strategy()
