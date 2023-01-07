import os
import random
import sys
from pathlib import Path
from typing import Literal, Optional

import seaborn as sns
import tensorflow as tf
from loguru import logger
from modin.config import Engine

from boiling_learning.app.constants import masters_path
from boiling_learning.descriptions import describe
from boiling_learning.lazy import LazyDescribed
from boiling_learning.utils.pathutils import resolve
from boiling_learning.utils.typeutils import typename


def configure(
    *,
    modin_engine: Literal['ray'],
    force_gpu_allow_growth: bool = False,
    use_xla: bool = False,
    mixed_precision_global_policy: str = 'float32',
    require_gpu: bool = False,
    nvidia_output: Optional[str] = None,
) -> LazyDescribed[tf.distribute.Strategy]:
    _configure_random_state()
    _configure_logger()
    _configure_gpu_growth(force_gpu_allow_growth)
    strategy: tf.distribute.Strategy = _configure_gpu(
        require_gpu=require_gpu, nvidia_output=nvidia_output
    )

    _configure_tensorflow(use_xla, mixed_precision_global_policy)
    _configure_modin_engine(modin_engine)

    _configure_seaborn()

    lazy_strategy = LazyDescribed.from_value_and_description(
        strategy,
        # type-ignore is necessary until the classes plugin works again
        typename(strategy),  # type: ignore[arg-type]
    )

    # type-ignore is necessary until the classes plugin works again
    strategy_description: str = describe(lazy_strategy)  # type: ignore[arg-type]
    logger.info(f'Using distribute strategy: {strategy_description}')
    return lazy_strategy


def _configure_random_state() -> None:
    random.seed(1997)


def _configure_logger() -> None:
    logger.remove()  # remove default logger configuration
    logger.add(sys.stderr, level='DEBUG')
    logger.add(str(_log_file_path()), level='DEBUG')

    logger.info('Initializing script')


def _log_file_path() -> Path:
    return resolve(masters_path() / 'logs' / '{time}.log', parents=True)


def _configure_gpu_growth(force_gpu_allow_growth: bool) -> None:
    if force_gpu_allow_growth:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def _configure_tensorflow(
    use_xla: bool = False,
    mixed_precision_global_policy: str = 'float32',
) -> None:
    tf.config.optimizer.set_jit(use_xla)
    tf.keras.mixed_precision.set_global_policy(mixed_precision_global_policy)


def _configure_modin_engine(modin_engine: Literal['ray']) -> None:
    Engine.put(modin_engine)


def _configure_seaborn() -> None:
    sns.set_style('whitegrid')
    sns.set(
        rc={
            'text.usetex': True,
            'font.size': 12.0,
            'font.family': 'serif',
            'font.serif': 'Computer Modern',
            'axes.titlesize': 'medium',
            'figure.titlesize': 'medium',
            'text.latex.preamble': '\n'.join(
                (
                    '\\usepackage{amsmath}',
                    '\\usepackage{amssymb}',
                    '\\usepackage[per-mode=symbol]{siunitx}',
                )
            ),
        },
        style='whitegrid',
    )


def _configure_gpu(
    require_gpu: bool = True,
    nvidia_output: Optional[str] = None,
) -> tf.distribute.Strategy:
    logger.info('Connecting to GPUs')

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
