from __future__ import annotations

from typing import Any, Callable, Generic, Tuple, TypeVar

from typing_extensions import Protocol

from boiling_learning.io import json
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.lazy import Lazy

_X_contra = TypeVar('_X_contra', contravariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_X = TypeVar('_X')
_Y = TypeVar('_Y')


class Transformer(Generic[_X, _Y]):
    def __init__(self, f: CallableWithFirst[_X, _Y], pack: Pack[Any, Any] = Pack()) -> None:
        self._call: Callable[[_X], _Y] = pack.rpartial(f)
        self.pack: Pack[Any, Any] = pack

    def __call__(self, arg: _X) -> _Y:
        return self._call(arg)

    def __describe__(self) -> json.JSONDataType:
        return json.serialize({'type': self.__class__.__name__, 'pack': self.pack})

    def __str__(self) -> str:
        args = ', '.join(str(arg) for arg in self.pack.args)
        kwargs = ', '.join(f'{key}={value}' for key, value in self.pack.kwargs.items())
        arguments = f'{args}, {kwargs}' if args and kwargs else args or kwargs
        return f'<{self.__class__.__name__} ({arguments})>'

    def __ror__(self, arg: _X) -> _Y:
        return _Transformed(arg, self)


class _Transformed(Lazy[_Y]):
    def __init__(self, arg: _X, transformer: Transformer[_X, _Y]) -> None:
        if isinstance(arg, _Transformed):
            super().__init__(lambda: transformer(arg()))
        else:
            super().__init__(lambda: transformer(arg))

        self._arg = arg
        self._transformer = transformer

    def __describe__(self) -> json.JSONDataType:
        return describe(self._pipe())

    def _pipe(self) -> Tuple[Any, ...]:
        if isinstance(self._arg, _Transformed):
            return (*self._arg._pipe(), self._transformer)  # pylint: disable=protected-access
        else:
            return (self._arg, self._transformer)


# the concept of mathematical operator as a function mapping a set to itself
Operator = Transformer[_X, _X]


@json.encode.instance(Transformer)
def _encode_transformer(instance: Transformer[Any, Any]) -> json.JSONDataType:
    return json.serialize(describe(instance))


class CallableWithFirst(Protocol[_X_contra, _Y_co]):
    def __call__(self, first: _X_contra, *args: Any, **kwargs: Any) -> _Y_co:
        ...
