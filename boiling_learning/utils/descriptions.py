from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, FrozenSet, Generic, List, Set, Tuple, Type, TypeVar, Union

from classes import AssociatedType, Supports, typeclass
from typing_extensions import Protocol, final, runtime_checkable

from boiling_learning.utils.dataclasses import asdict, is_dataclass_instance
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


class _ListOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, list) and all(describe.supports(item) for item in instance)


class ListOfDescribable(
    List[Supports[Describable[_Description]]],
    Generic[_Description],
    metaclass=_ListOfDescribableMeta,
):
    ...


@describe.instance(delegate=ListOfDescribable)
def _describe_list(instance: ListOfDescribable[_Description]) -> List[_Description]:
    return [describe(item) for item in instance]


class _TupleOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(describe.supports(item) for item in instance)


class TupleOfDescribable(
    Tuple[Supports[Describable[_Description]], ...],
    Generic[_Description],
    metaclass=_TupleOfDescribableMeta,
):
    ...


@describe.instance(delegate=TupleOfDescribable)
def _describe_tuple(instance: TupleOfDescribable[_Description]) -> Tuple[_Description, ...]:
    return tuple(describe(item) for item in instance)


class _DictOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and describe.supports(value) for key, value in instance.items()
        )


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
        return is_dataclass_instance(instance) and describe.supports(asdict(instance))


class DataclassOfDescribableFields(
    Generic[_Description],
    metaclass=_DataclassOfDescribableFieldsMeta,
):
    ...


@describe.instance(delegate=DataclassOfDescribableFields)
def _describe_dataclass(
    instance: DataclassOfDescribableFields[_Description],
) -> Dict[str, _Description]:
    return describe(asdict(instance))


@describe.instance(Fraction)
def _describe_fraction(instance: Fraction) -> Tuple[int, int]:
    return instance.numerator, instance.denominator


class _FrozenDictOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, frozendict) and describe.supports(dict(instance))


class _FrozenDictMeta(_FrozenDictOfDescribableMeta, type(frozendict)):
    pass


class FrozenDictOfDescribable(
    frozendict[str, Supports[Describable[_Description]]],
    Generic[_Description],
    metaclass=_FrozenDictMeta,
):
    ...


@describe.instance(delegate=FrozenDictOfDescribable)
def _describe_frozendict(
    instance: FrozenDictOfDescribable[_Description],
) -> Dict[str, _Description]:
    return describe(dict(instance))


class _SetOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, set) and all(describe.supports(item) for item in instance)


class SetOfDescribable(
    Set[Supports[Describable[_Description]]],
    Generic[_Description],
    metaclass=_SetOfDescribableMeta,
):
    ...


@describe.instance(delegate=SetOfDescribable)
def _describe_set(instance: SetOfDescribable[_Description]) -> Set[_Description]:
    return {describe(item) for item in instance}


class _FrozenSetOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, frozenset) and all(
            describe.supports(item) for item in instance
        )


class FrozenSetOfDescribable(
    FrozenSet[Supports[Describable[_Description]]],
    Generic[_Description],
    metaclass=_FrozenSetOfDescribableMeta,
):
    ...


@describe.instance(delegate=FrozenSetOfDescribable)
def _describe_frozenset(instance: FrozenSetOfDescribable[_Description]) -> FrozenSet[_Description]:
    return frozenset(describe(item) for item in instance)


@describe.instance(type)
def _describe_type(instance: _AnyType) -> _AnyType:
    return instance
