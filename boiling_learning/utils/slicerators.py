from typing import TYPE_CHECKING, Generic, TypeVar

from slicerator import Slicerator as _Slicerator

_T = TypeVar('_T')


if TYPE_CHECKING:
    Slicerator = _Slicerator
else:

    class Slicerator(_Slicerator, Generic[_T]):
        pass
