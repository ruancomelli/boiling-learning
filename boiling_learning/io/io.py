from contextlib import nullcontext
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

from tensorflow.keras.models import load_model

from boiling_learning.utils import PathLike

_T = TypeVar('_T')
SaverFunction = Callable[[_T, PathLike], Any]
LoaderFunction = Callable[[PathLike], _T]


class DatasetTriplet(Tuple[_T, Optional[_T], _T], Generic[_T]):
    pass


# TODO: replicate the `strategy` behavior in the new model loading functionality
def load_keras_model(path: PathLike, strategy=None, **kwargs):
    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        return load_model(path, **kwargs)
