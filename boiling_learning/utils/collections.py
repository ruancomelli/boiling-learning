from __future__ import annotations

from contextlib import suppress
from itertools import chain
from typing import (
    Callable,
    Generic,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MutableSet,
    Optional,
    TypeVar,
    Union,
    ValuesView,
    overload,
)

_Any = TypeVar('_Any')
_Key = TypeVar('_Key', bound=Hashable)
_Value = TypeVar('_Value')


class KeyedSet(MutableSet[_Value], Generic[_Key, _Value]):
    def __init__(self, key: Callable[[_Value], _Key], iterable: Iterable[_Value] = ()) -> None:
        self.__key = key
        self.__data = {self.__key(element): element for element in iterable}

    # ------------------------------------------------------------------------
    # Set-like operations
    # ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.__data)

    def __contains__(self, element: object) -> bool:
        return element in self.values()

    def __iter__(self) -> Iterator[_Value]:
        return iter(self.values())

    def add(self, *values: _Value) -> None:
        for value in values:
            self.__data[self.__key(value)] = value

    def remove(self, *values: _Value) -> None:
        for value in values:
            del self[self.__key(value)]

    def discard(self, *values: _Value) -> None:
        for value in values:
            with suppress(KeyError):
                self.remove(value)

    def isdisjoint(self, other: Iterable[_Value]) -> bool:
        return frozenset(self.keys()).isdisjoint(map(self.__key, other))

    def issubset(self, other: Iterable[_Value]) -> bool:
        return frozenset(self.keys()).issubset(map(self.__key, other))

    def issuperset(self, other: Iterable[_Value]) -> bool:
        return frozenset(self.keys()).issuperset(map(self.__key, other))

    def union(self, *others: Iterable[_Value]) -> KeyedSet[_Key, _Value]:
        return KeyedSet(self.__key, chain(self, *others))

    def update(self, *others: Iterable[_Value]) -> None:
        for other in others:
            for item in other:
                self.add(item)

    def clear(self) -> None:
        self.__data.clear()

    # TODO: implement other operations as in the standard docs:
    # <https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset>

    # ------------------------------------------------------------------------
    # Dict-like operations
    # ------------------------------------------------------------------------
    def __getitem__(self, key: _Key) -> _Value:
        return self.__data[key]

    def __delitem__(self, key: _Key) -> None:
        del self.__data[key]

    def keys(self) -> KeysView[_Key]:
        return self.__data.keys()

    def values(self) -> ValuesView[_Value]:
        return self.__data.values()

    def items(self) -> ItemsView[_Key, _Value]:
        return self.__data.items()

    @overload
    def get(self, key: _Key, default: _Any) -> Union[_Value, _Any]:
        ...

    @overload
    def get(self, key: _Key, default: None = None) -> Union[_Value, None]:
        ...

    def get(self, key: _Key, default: Optional[_Any] = None) -> Union[_Value, Optional[_Any]]:
        return self.__data.get(key, default)
