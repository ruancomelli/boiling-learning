from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import Concatenate, ParamSpec

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.lazy import Lazy, LazyTransform
from boiling_learning.utils.functional import Pack

_X = TypeVar('_X')
_Y = TypeVar('_Y')
_P = ParamSpec('_P')


class Transformer(Generic[_X, _Y]):
    def __init__(
        self,
        function: Callable[Concatenate[_X, _P], _Y],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        self.function = function
        self.pack = Pack(args, kwargs)

    def __call__(self, arg: _X) -> _Y:
        return self.function(arg, *self.pack.args, **self.pack.kwargs)

    def __describe__(self) -> json.JSONEncodable:
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


@json.encode.instance(Transformer)
def _encode_transformer(instance: Transformer[Any, Any]) -> json.JSONDataType:
    return json.serialize(describe(instance))


def wrap_as_partial_transformer(
    function: Callable[Concatenate[_X, _P], _Y]
) -> Callable[_P, Transformer[_X, _Y]]:
    return partial(Transformer, function)
