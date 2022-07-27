from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Generic, TypeVar

__all__ = ('Lazy', 'LazyCallable')

_T = TypeVar('_T')


class Lazy(Generic[_T]):
    def __init__(self, creator: Callable[[], _T]) -> None:
        self._creator: Callable[[], _T] = creator

    @lru_cache(maxsize=1)
    def __call__(self) -> _T:
        return self._creator()

    @classmethod
    def from_value(cls, value: _T) -> Lazy[_T]:
        return Lazy(lambda: value)


class LazyCallable(Generic[_T]):
    def __init__(self, call: Callable[..., _T]) -> None:
        self._call: Callable[..., _T] = call

    def __call__(self, *args: Any, **kwargs: Any) -> Lazy[_T]:
        return Lazy(partial(self._call, *args, **kwargs))
