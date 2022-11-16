import os
from typing import Literal

import tensorflow as tf
from modin.config import Engine


def configure(
    *,
    modin_engine: Literal['ray'],
    force_gpu_allow_growth: bool = False,
    use_xla: bool = False,
    mixed_precision_global_policy: str = 'float32',
) -> None:
    if force_gpu_allow_growth:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tf.config.optimizer.set_jit(use_xla)
    tf.keras.mixed_precision.set_global_policy(mixed_precision_global_policy)

    Engine.put(modin_engine)
