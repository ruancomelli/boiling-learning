from typing import Any, Tuple, Type, TypeVar

import typeguard
from typing_extensions import ParamSpec, Protocol

_T = TypeVar('_T')
_X_contra = TypeVar('_X_contra', contravariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_P = ParamSpec('_P')
Many = Tuple[_T, ...]


def typechecks(obj: Any, type_: Type[Any]) -> bool:
    try:
        typeguard.check_type('', obj, type_)
        return True
    except TypeError:
        return False


def typename(obj: Any) -> str:
    return type(obj).__name__


class SupportsLessThan(Protocol):
    def __lt__(self, other: Any) -> bool:
        pass


SupportsLessThanT = TypeVar('SupportsLessThanT', bound=SupportsLessThan)


class CallableWithFirst(Protocol[_X_contra, _P, _Y_co]):
    def __call__(self, x: _X_contra, *args: _P.args, **kwargs: _P.kwargs) -> _Y_co:
        ...
