from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, TypeVar, Union

_Key = TypeVar('_Key')
_OtherKey = TypeVar('_OtherKey')
_Value = TypeVar('_Value')
_OtherValue = TypeVar('_OtherValue')


class FrozenDict(Dict[_Key, _Value], Mapping[_Key, _Value]):
    '''
    Source: https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
    Some modifications based on: https://www.python.org/dev/peps/pep-0603/
    '''

    @lru_cache(maxsize=1)
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))

    # makes (deep)copy alot more efficient
    def __copy__(self) -> FrozenDict[_Key, _Value]:
        return self

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> FrozenDict[_Key, _Value]:
        if memo is not None:
            memo[id(self)] = self
        return self

    def _immutable(self, *args, **kws):
        raise TypeError('cannot change object - object is immutable')

    def union(
        self,
        mapping: Optional[Mapping[_OtherKey, _OtherValue]],
        **kw: _OtherValue,
    ) -> FrozenDict[Union[str, _Key, _OtherKey], Union[_Value, _OtherValue]]:
        return FrozenDict({**self, **mapping}, **kw)

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
