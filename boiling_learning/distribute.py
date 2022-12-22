from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import tensorflow as tf

from boiling_learning.lazy import Lazy


@contextmanager
def strategy_scope(strategy: Lazy[tf.distribute.Strategy] | None) -> Iterator[None]:
    context = strategy().scope() if strategy is not None else nullcontext()

    with context:
        yield
