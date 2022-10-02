from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Generic, Tuple, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec

from boiling_learning.describe.describers import describe
from boiling_learning.io import json

__all__ = ('Lazy', 'LazyCallable', 'LazyTransform')

_T = TypeVar('_T')
_S = TypeVar('_S')
_P = ParamSpec('_P')


class Lazy(Generic[_T]):
    def __init__(self, creator: Callable[[], _T]) -> None:
        self._creator = lru_cache(maxsize=1)(creator)

    def __call__(self) -> _T:
        return self._creator()

    @classmethod
    def from_value(cls, value: _T) -> Lazy[_T]:
        return Lazy(lambda: value)

    def __describe__(self) -> json.JSONDataType:
        return describe(self())


class LazyDescribed(Lazy[_T]):
    def __init__(self, value: Lazy[_T], description: json.JSONDataType) -> None:
        super().__init__(value)
        self._description = description

    def __describe__(self) -> json.JSONDataType:
        return describe(self._description)

    @classmethod
    def from_value_and_description(
        cls, value: _T, description: json.JSONDataType
    ) -> LazyDescribed[_T]:
        return LazyDescribed(Lazy.from_value(value), description)


class LazyCallable(Generic[_P, _T]):
    def __init__(self, call: Callable[_P, _T]) -> None:
        self._call = call

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> Lazy[_T]:
        return Lazy(partial(self._call, *args, **kwargs))


class LazyTransform(Lazy[_S]):
    def __init__(self, arg: Lazy[_T], transform: Callable[[_T], _S]) -> None:
        self._arg = arg
        self._transform = transform
        super().__init__(self._eval)

    def _eval(self) -> _S:
        return self._transform(self._arg())

    def __describe__(self) -> json.JSONDataType:
        return describe(self._pipeline())

    def _pipeline(self) -> Tuple[Any, ...]:
        return (
            (*self._arg._pipeline(), self._transform)  # pylint: disable=protected-access
            if isinstance(self._arg, LazyTransform)
            else (self._arg, self._transform)
        )


def eager(
    function: Callable[Concatenate[_T, _P], _S]
) -> Callable[Concatenate[Union[_T, Lazy[_T]], _P], _S]:
    def _wrapped(first: Union[_T, Lazy[_T]], *args: _P.args, **kwargs: _P.kwargs) -> _S:
        return (
            function(first(), *args, **kwargs)
            if isinstance(first, Lazy)
            else function(first, *args, **kwargs)
        )

    return _wrapped
