from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable, Generic, TypeVar

from typing_extensions import ParamSpec

__all__ = ('Lazy', 'LazyCallable')

_T = TypeVar('_T')
_P = ParamSpec('_P')


class Lazy(Generic[_T]):
    def __init__(self, creator: Callable[[], _T]) -> None:
        self._creator = lru_cache(maxsize=1)(creator)

    def __call__(self) -> _T:
        return self._creator()

    @classmethod
    def from_value(cls, value: _T) -> Lazy[_T]:
        return Lazy(lambda: value)


class LazyCallable(Generic[_P, _T]):
    def __init__(self, call: Callable[_P, _T]) -> None:
        self._call = call

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Lazy[_T]:
        return Lazy(partial(self._call, *args, **kwargs))
