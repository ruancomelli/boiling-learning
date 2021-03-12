from typing import Mapping, TypeVar

_Key = TypeVar('_Key')
_Value = TypeVar('_Value')

class FrozenDict(dict, Mapping[_Key, _Value]):
    '''
    Source: https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
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

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
