from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, FrozenSet, Generic, List, Set, Tuple, Type, TypeVar, Union

from classes import AssociatedType, Supports, typeclass
from typing_extensions import Protocol, final, runtime_checkable

from boiling_learning.utils.dataclasses import is_dataclass_instance, shallow_asdict
from boiling_learning.utils.frozendict import frozendict

_AnyType = TypeVar('_AnyType', bound=Type[Any])
_Description = TypeVar('_Description')
_Description_co = TypeVar('_Description_co', covariant=True)


@runtime_checkable
class HasDescribe(Protocol[_Description_co]):
    def __describe__(self) -> _Description_co:
        ...


@final
class Describable(AssociatedType[_Description]):
    ...


@typeclass(Describable)
def describe(instance: Supports[Describable[_Description]]) -> _Description:
    '''Return a JSON description of an object.'''


_BasicType = TypeVar('_BasicType', bound=Union[None, bool, int, str, float, Path])


@describe.instance(None)
@describe.instance(bool)
@describe.instance(int)
@describe.instance(str)
@describe.instance(float)
@describe.instance(Path)
def _describe_basics(instance: _BasicType) -> _BasicType:
    return instance


@describe.instance(protocol=HasDescribe)
def _describe_has_describe(instance: HasDescribe[_Description]) -> _Description:
    return instance.__describe__()


@describe.instance(list)
def _describe_list(instance: List[Supports[Describable[_Description]]]) -> List[_Description]:
    return [describe(item) for item in instance]


@describe.instance(tuple)
def _describe_tuple(
    instance: Tuple[Supports[Describable[_Description]], ...]
) -> Tuple[_Description, ...]:
    return tuple(describe(item) for item in instance)


class _DictOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(isinstance(key, str) for key in instance)


class DictOfDescribable(
    Dict[str, Supports[Describable[_Description]]],
    Generic[_Description],
    metaclass=_DictOfDescribableMeta,
):
    ...


@describe.instance(delegate=DictOfDescribable)
def _describe_dict(instance: DictOfDescribable[_Description]) -> Dict[str, _Description]:
    return {key: describe(value) for key, value in instance.items()}


class _DataclassOfDescribableFieldsMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return is_dataclass_instance(instance) and describe.supports(shallow_asdict(instance))


class DataclassOfDescribableFields(
    Generic[_Description],
    metaclass=_DataclassOfDescribableFieldsMeta,
):
    ...


@describe.instance(delegate=DataclassOfDescribableFields)
def _describe_dataclass(
    instance: DataclassOfDescribableFields[_Description],
) -> Dict[str, _Description]:
    return describe(shallow_asdict(instance))


@describe.instance(Fraction)
def _describe_fraction(instance: Fraction) -> Tuple[int, int]:
    return instance.numerator, instance.denominator


@describe.instance(timedelta)
def _describe_timedelta(instance: timedelta) -> float:
    return instance.total_seconds()


@describe.instance(frozendict)
def _describe_frozendict(
    instance: frozendict[str, Supports[Describable[_Description]]],
) -> Dict[str, _Description]:
    return describe(dict(instance))


@describe.instance(set)
def _describe_set(instance: Set[Supports[Describable[_Description]]]) -> Set[_Description]:
    return {describe(item) for item in instance}


@describe.instance(frozenset)
def _describe_frozenset(
    instance: FrozenSet[Supports[Describable[_Description]]],
) -> FrozenSet[_Description]:
    return frozenset(describe(item) for item in instance)


@describe.instance(type)
def _describe_type(instance: _AnyType) -> _AnyType:
    return instance
