from __future__ import annotations

from functools import lru_cache, partial, wraps
from typing import Any, Callable, Generic, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec

from boiling_learning.descriptions import Describable, describe
from boiling_learning.io import json
from boiling_learning.utils.functional import Pack

__all__ = ('Lazy', 'LazyCallable', 'LazyTransform')

_T = TypeVar('_T')
_S = TypeVar('_S')
_P = ParamSpec('_P')
_Desc = TypeVar('_Desc', bound=Describable[json.JSONDataType])


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
    def __init__(self, value: Lazy[_T], description: Describable[json.JSONDataType]) -> None:
        super().__init__(value)
        self._description = describe(description)

    def __describe__(self) -> json.JSONDataType:
        return self._description

    @classmethod
    def from_value_and_description(
        cls, value: _T, description: Describable[json.JSONDataType]
    ) -> LazyDescribed[_T]:
        return LazyDescribed(Lazy.from_value(value), description)

    @classmethod
    def from_describable(cls, value: _Desc) -> Lazy[_Desc]:
        return cls.from_value_and_description(value, value)

    @classmethod
    def from_constructor(
        cls, constructor: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> LazyDescribed[_T]:
        return LazyDescribed(
            LazyCallable(constructor)(*args, **kwargs),
            (constructor, Pack(args, kwargs)),
        )

    @classmethod
    def from_list(cls, items: list[LazyDescribed[_T]]) -> LazyDescribed[list[_T]]:
        return LazyDescribed(Lazy(lambda: [item() for item in items]), items)


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

    def _pipeline(self) -> tuple[Any, ...]:
        return (
            (*self._arg._pipeline(), self._transform)  # pylint: disable=protected-access
            if isinstance(self._arg, LazyTransform)
            else (self._arg, self._transform)
        )


def eager(
    function: Callable[Concatenate[_T, _P], _S]
) -> Callable[Concatenate[Union[_T, Lazy[_T]], _P], _S]:
    @wraps(function)
    def _wrapped(first: Union[_T, Lazy[_T]], *args: _P.args, **kwargs: _P.kwargs) -> _S:
        return (
            function(first(), *args, **kwargs)
            if isinstance(first, Lazy)
            else function(first, *args, **kwargs)
        )

    return _wrapped
