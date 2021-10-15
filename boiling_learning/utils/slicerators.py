from typing import Generic, TypeVar

from slicerator import Slicerator as _Slicerator

_T = TypeVar('_T')


class Slicerator(_Slicerator, Generic[_T]):
    pass
