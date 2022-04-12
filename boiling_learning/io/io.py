from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

from boiling_learning.utils import PathLike

_T = TypeVar('_T')
SaverFunction = Callable[[_T, PathLike], Any]
LoaderFunction = Callable[[PathLike], _T]


class DatasetTriplet(Tuple[_T, Optional[_T], _T], Generic[_T]):
    pass
