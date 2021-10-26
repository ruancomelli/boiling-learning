from __future__ import annotations

from typing import Any, Callable, Generic, Sequence, TypeVar

import funcy

from boiling_learning.io.storage import json_encode
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import JSONDataType

_In = TypeVar('_In')
_Out = TypeVar('_Out')


class NamedFunction(Generic[_Out]):
    def __init__(self, name: str, call: Callable[..., _Out]) -> None:
        self.__name: str = name
        self.__call: Callable[..., _Out] = call

    @property
    def name(self) -> str:
        return self.__name

    def __call__(self, *args: Any, **kwargs: Any) -> _Out:
        return self.__call(*args, **kwargs)


class NamedFunctional(Generic[_Out]):
    def __init__(
        self, name: str, callable: Callable[..., _Out], pack: Pack = Pack()
    ) -> None:
        self.__name: str = name
        self.__call: Callable[..., _Out] = pack.rpartial(callable)
        self.__desc: Pack = pack

    @property
    def name(self) -> str:
        return self.__name

    @property
    def desc(self) -> Pack:
        return self.__desc

    def __call__(self, *args: Any, **kwargs: Any) -> _Out:
        return self.__call(*args, **kwargs)


class Creator(NamedFunctional[_Out], Generic[_Out]):
    def __init__(
        self, name: str, callable: Callable[..., _Out], pack: Pack = Pack()
    ) -> None:
        super().__init__(name, callable, pack)

    def __call__(self) -> _Out:
        return super().__call__()


class Transformer(NamedFunctional[_Out], Generic[_In, _Out]):
    def __init__(
        self, name: str, callable: Callable[..., _Out], pack: Pack = Pack()
    ) -> None:
        super().__init__(name, callable, pack)

    def __call__(self, arg: _In) -> _Out:
        return super().__call__(arg)


class FunctionalPipeline(Sequence[NamedFunctional]):
    def __init__(self, *funcs: NamedFunctional) -> None:
        self._pipe = funcs

    def __getitem__(self, index: int) -> NamedFunctional:
        return self._pipe[index]

    def __len__(self) -> int:
        return len(self._pipe)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return funcy.rcompose(*self)(*args, **kwds)

    def __add__(self, other: FunctionalPipeline) -> FunctionalPipeline:
        return FunctionalPipeline(*self, *other)

    def __json_encode__(self) -> JSONDataType:
        return [func.__json_encode__() for func in self]


# class FunctionRegistry(dict, Dict[str, NamedFunctional]):
#     def __init__(self, )


@json_encode.dispatch
def json_encode(obj: NamedFunctional):
    return {'name': obj.name, 'desc': json_encode(obj.desc)}
