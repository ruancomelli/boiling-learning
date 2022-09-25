from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import Concatenate, ParamSpec, Protocol

from boiling_learning.describe.describers import describe
from boiling_learning.io import json
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.lazy import Lazy, LazyTransform

_X_contra = TypeVar('_X_contra', contravariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_X = TypeVar('_X')
_Y = TypeVar('_Y')
_P = ParamSpec('_P')


class Transformer(Generic[_X, _Y]):
    def __init__(self, function: CallableWithFirst[_X, _Y], pack: Pack[Any, Any] = Pack()) -> None:
        self.function = function
        self.pack = pack

    def __call__(self, arg: _X) -> _Y:
        return self.function(arg, *self.pack.args, **self.pack.kwargs)

    def __describe__(self) -> json.JSONDataType:
        return {
            'type': type(self),
            'function': self.function,
            'pack': self.pack,
        }

    def __str__(self) -> str:
        arguments = chain(
            (str(arg) for arg in self.pack.args),
            (f'{key}={value}' for key, value in self.pack.kwargs.items()),
        )
        return f'<{self.__class__.__name__} ({", ".join(arguments)})>'

    def __ror__(self, arg: Lazy[_X]) -> LazyTransform[_Y]:
        return LazyTransform(arg, self)


# the concept of mathematical operator as a function mapping a set to itself
Operator = Transformer[_X, _X]


@json.encode.instance(Transformer)
def _encode_transformer(instance: Transformer[Any, Any]) -> json.JSONDataType:
    return json.serialize(describe(instance))


def wrap_as_partial_transformer(
    function: Callable[Concatenate[_X, _P], _Y]
) -> Callable[_P, Transformer[_X, _Y]]:
    def _partial_transformer(*args: _P.args, **kwargs: _P.kwargs) -> Transformer[_X, _Y]:
        return Transformer(function, Pack(args, kwargs))

    return _partial_transformer


class CallableWithFirst(Protocol[_X_contra, _Y_co]):
    def __call__(self, first: _X_contra, *args: Any, **kwargs: Any) -> _Y_co:
        ...
