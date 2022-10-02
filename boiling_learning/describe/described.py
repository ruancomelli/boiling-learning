from __future__ import annotations

from typing import Generic, List, Tuple, Type, TypeVar

from boiling_learning.describe.describers import describe
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.sentinels import EMPTY, Emptiable

_Any = TypeVar('_Any')
_AnyT = TypeVar('_AnyT')
_AnyS = TypeVar('_AnyS')
_Description = TypeVar('_Description')
_DescribedConstructedObject = Tuple[str, Pack[_AnyT, _AnyS]]


class Described(Generic[_Any, _Description]):
    def __init__(self, value: _Any, description: Emptiable[_Description] = EMPTY) -> None:
        self._value = value
        self.description = describe(description if description is not EMPTY else value)

    def __call__(self) -> _Any:
        return self._value

    def __describe__(self) -> _Description:
        return self.description

    @classmethod
    def from_constructor(
        cls, type_: Type[_Any], params: Pack[_AnyT, _AnyS]
    ) -> Described[_Any, _DescribedConstructedObject[_AnyT, _AnyS]]:
        return cls(params.feed(type_), describe((type_, params)))

    @classmethod
    def from_list(
        cls, described: List[Described[_Any, _Description]]
    ) -> Described[List[_Any], List[_Description]]:
        values = [item() for item in described]
        return cls(values, describe(described))
