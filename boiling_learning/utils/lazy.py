from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, TypeVar

import funcy
from lazy import lazy as lazy_property

__all__ = ('Lazy', 'LazyCallable', 'lazy_property')

_T = TypeVar('_T')
_S = TypeVar('_S')


class Lazy(Generic[_T]):
    def __init__(self, creator: Callable[[], _T]) -> None:
        self._creator: Callable[[], _T] = creator

    @lazy_property
    def __value(self) -> _T:
        return self._creator()

    def __call__(self) -> _T:
        return self.__value

    @classmethod
    def from_value(self, value: _T) -> Lazy[_T]:
        return Lazy(lambda: value)


class LazyCallable(Generic[_T]):
    def __init__(self, call: Callable[..., _T]) -> None:
        self._call: Callable[..., _T] = call

    def __call__(self, *args: Any, **kwargs: Any) -> Lazy[_T]:
        return Lazy(partial(self._call, *args, **kwargs))

    def __rmatmul__(self, other: Callable[[_T], _S]) -> LazyCallable[_S]:
        if callable(other):
            return LazyCallable(funcy.compose(other, self._call))
        return NotImplemented
