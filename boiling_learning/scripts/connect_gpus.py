from typing import Optional

import tensorflow as tf
from loguru import logger


def main(
    require_gpu: bool = True,
    nvidia_output: Optional[str] = None,
) -> tf.distribute.Strategy:
    logger.info('Connecting to GPUs')

    # See <https://www.tensorflow.org/xla/tutorials/autoclustering_xla>
    tf.config.optimizer.set_jit(True)  # Enable XLA.

    gpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices('GPU')

    logger.info(f'Available GPUs: {gpus}')

    if gpus:
        logger.info('Connected to GPUs.')

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce(num_packs=0)
        )

    if nvidia_output is not None and nvidia_output.startswith('GPU'):
        raise RuntimeError(
            'command *nvidia-smi -L* found GPUs, but TensorFlow could not connect to them.'
        )

    if require_gpu:
        raise RuntimeError('No GPUs connected.')

    logger.info('No GPUs connected.')
    return tf.distribute.get_strategy()
