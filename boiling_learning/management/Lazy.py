from __future__ import annotations

from typing import Callable, Generic, TypeVar

from lazy import lazy

_T = TypeVar('_T')


class Lazy(Generic[_T]):
    def __init__(self, creator: Callable[[], _T]) -> None:
        self._creator: Callable[[], _T] = creator

    @lazy
    def value(self) -> _T:
        return self._creator()

    @classmethod
    def from_value(self, value: _T) -> Lazy[_T]:
        return Lazy(lambda: value)
