from typing import Mapping, Optional, TypeVar, Union

_Key = TypeVar('_Key')
_OtherKey = TypeVar('_OtherKey')
_Value = TypeVar('_Value')
_OtherValue = TypeVar('_OtherValue')


class FrozenDict(dict, Mapping[_Key, _Value]):
    '''
    Source: https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
    Some modifications based on: https://www.python.org/dev/peps/pep-0603/
    '''
    def __init__(self, *args, **kwargs):
        self._hash = None
        super().__init__(*args, **kwargs)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(sorted(self.items())))
        return self._hash

    # makes (deep)copy alot more efficient
    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        if memo is not None:
            memo[id(self)] = self
        return self

    def _immutable(self, *args, **kws):
        raise TypeError('cannot change object - object is immutable')

    def union(
            self,
            mapping: Optional[Mapping[_OtherKey, _OtherValue]],
            **kw: _OtherValue
    ) -> 'FrozenDict[Union[str, _Key, _OtherKey], Union[_Value, _OtherValue]]':
        return FrozenDict({**self, **mapping}, **kw)

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
