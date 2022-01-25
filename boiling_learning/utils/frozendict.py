import sys
from typing import Generic, TypeVar

from frozendict import frozendict as _frozendict

_T = TypeVar('_T')
_S = TypeVar('_S')


__all__ = ('frozendict',)

if sys.version_info[:3] >= (3, 9, 0):

    class frozendict(_frozendict):
        pass

else:

    class frozendict(_frozendict, Generic[_T, _S]):
        pass
