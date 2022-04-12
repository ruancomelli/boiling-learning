from typing import Any, Callable, TypeVar

from boiling_learning.utils.utils import PathLike

_T = TypeVar('_T')
SaverFunction = Callable[[_T, PathLike], Any]
LoaderFunction = Callable[[PathLike], _T]
