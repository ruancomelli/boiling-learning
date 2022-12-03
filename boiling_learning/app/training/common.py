import functools
from typing import Literal

import tensorflow as tf

from boiling_learning.app.paths import analyses_path
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import CachedFunction, Cacher
from boiling_learning.model.training import get_fit_model, load_with_strategy


@functools.cache
def cached_fit_model_function(
    experiment: Literal['boiling1d', 'condensation'],
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> CachedFunction:
    return CachedFunction(
        get_fit_model,
        Cacher(
            allocator=JSONAllocator(analyses_path() / 'models' / experiment),
            exceptions=(FileNotFoundError, NotADirectoryError, tf.errors.OpError),
            loader=load_with_strategy(strategy),
        ),
    )
