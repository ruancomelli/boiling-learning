import sys

__all__ = ('frozendict',)

if sys.version_info[:3] >= (3, 9, 0):
    from frozendict import frozendict
else:
    from typing import Generic, TypeVar

    from frozendict import frozendict as _frozendict

    _T = TypeVar('_T')
    _S = TypeVar('_S')

    class frozendict(_frozendict, Generic[_T, _S]):
        pass
