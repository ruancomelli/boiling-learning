import os
import sys
from typing import Literal

import tensorflow as tf
from loguru import logger
from modin.config import Engine

from boiling_learning.app.constants import MASTERS_PATH
from boiling_learning.utils.pathutils import resolve

LOG_FILE_PATH = resolve(MASTERS_PATH / 'logs' / '{time}.log', parents=True)


def configure(
    *,
    modin_engine: Literal['ray'],
    force_gpu_allow_growth: bool = False,
    use_xla: bool = False,
    mixed_precision_global_policy: str = 'float32',
) -> None:
    _configure_gpu_growth(force_gpu_allow_growth)
    _configure_tensorflow(use_xla, mixed_precision_global_policy)
    _configure_modin_engine(modin_engine)
    _configure_logger()


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


def _configure_logger() -> None:
    logger.remove()  # remove default logger configuration
    logger.add(sys.stderr, level='DEBUG')
    logger.add(str(LOG_FILE_PATH), level='DEBUG')

    logger.info('Initializing script')
