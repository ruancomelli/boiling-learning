from typing import Generic, TypeVar

from dataclassy import dataclass

_Any = TypeVar('_Any')
_Description = TypeVar('_Description')


@dataclass(frozen=True, slots=True)
class Described(Generic[_Any, _Description]):
    value: _Any
    description: _Description

    def __describe__(self) -> _Description:
        return self.description
