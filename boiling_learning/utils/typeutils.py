from typing import Any, TypeVar

from typing_extensions import ParamSpec, Protocol

_X_contra = TypeVar('_X_contra', contravariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_P = ParamSpec('_P')


def typename(obj: Any) -> str:
    return type(obj).__name__


class CallableWithFirst(Protocol[_X_contra, _P, _Y_co]):
    def __call__(self, x: _X_contra, *args: _P.args, **kwargs: _P.kwargs) -> _Y_co:
        ...
