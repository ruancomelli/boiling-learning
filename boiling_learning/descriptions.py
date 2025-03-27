from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from types import FunctionType
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    final,
    runtime_checkable,
)

from classes import AssociatedType, Supports, typeclass
from frozendict import frozendict  # type: ignore[attr-defined]

from boiling_learning.dataclasses import is_dataclass_instance, shallow_asdict

_AnyType = TypeVar("_AnyType", bound=type[Any])
_Description = TypeVar("_Description")
_Description_co = TypeVar("_Description_co", covariant=True)


@runtime_checkable
class HasDescribe(Protocol[_Description_co]):
    def __describe__(self) -> _Description_co: ...


@final
class Describable(AssociatedType[_Description]): ...


@typeclass(Describable)
def describe(instance: Supports[Describable[_Description]]) -> _Description:
    """Return a JSON description of an object."""


_BasicType = TypeVar("_BasicType", bound=None | bool | int | str | float | Path)


@describe.instance(None)
@describe.instance(bool)
@describe.instance(int)
@describe.instance(str)
@describe.instance(float)
@describe.instance(Path)
@describe.instance(FunctionType)
def _describe_basics(instance: _BasicType) -> _BasicType:
    return instance


@describe.instance(protocol=HasDescribe)
def _describe_has_describe(instance: HasDescribe[_Description]) -> _Description:
    return instance.__describe__()


@describe.instance(list)
def _describe_list(
    instance: list[Supports[Describable[_Description]]],
) -> list[_Description]:
    return [describe(item) for item in instance]


@describe.instance(tuple)
def _describe_tuple(
    instance: tuple[Supports[Describable[_Description]], ...],
) -> tuple[_Description, ...]:
    return tuple(describe(item) for item in instance)


@describe.instance(dict)
def _describe_dict(
    instance: dict[str, Supports[Describable[_Description]]],
) -> dict[str, _Description]:
    return {key: describe(value) for key, value in instance.items()}


class _DataclassOfDescribableFieldsMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return is_dataclass_instance(instance) and describe.supports(
            shallow_asdict(instance)
        )


class DataclassOfDescribableFields(
    Generic[_Description],
    metaclass=_DataclassOfDescribableFieldsMeta,
): ...


@describe.instance(delegate=DataclassOfDescribableFields)
def _describe_dataclass(
    instance: DataclassOfDescribableFields[_Description],
) -> dict[str, _Description]:
    return describe(shallow_asdict(instance))


@describe.instance(Fraction)
def _describe_fraction(instance: Fraction) -> tuple[int, int]:
    return instance.numerator, instance.denominator


@describe.instance(timedelta)
def _describe_timedelta(instance: timedelta) -> float:
    return instance.total_seconds()


@describe.instance(frozendict)
def _describe_frozendict(
    instance: frozendict[str, Supports[Describable[_Description]]],
) -> dict[str, _Description]:
    return describe(dict(instance))


@describe.instance(set)
def _describe_set(
    instance: set[Supports[Describable[_Description]]],
) -> set[_Description]:
    return {describe(item) for item in instance}


@describe.instance(frozenset)
def _describe_frozenset(
    instance: frozenset[Supports[Describable[_Description]]],
) -> frozenset[_Description]:
    return frozenset(describe(item) for item in instance)


@describe.instance(type)
def _describe_type(instance: _AnyType) -> _AnyType:
    return instance
