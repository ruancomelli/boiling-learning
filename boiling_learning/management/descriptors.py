from typing import Any, Dict, List, TypeVar, Union

from classes import AssociatedType as _AssociatedType
from classes import Supports
from classes import typeclass as _typeclass
from typing_extensions import Protocol, runtime_checkable

from boiling_learning.utils.utils import JSONDataType

_T = TypeVar('_T')


@runtime_checkable
class HasDescribe(Protocol):
    def __describe__(self) -> JSONDataType:
        ...


class Describable(_AssociatedType):
    ...


@_typeclass(Describable)
def describe(instance: Any) -> JSONDataType:
    '''Return a JSON description of an object.'''


BasicTypes = Union[None, bool, int, str, float]
_BasicType = TypeVar('_BasicType', bound=BasicTypes)


@describe.instance(None)
@describe.instance(bool)
@describe.instance(int)
@describe.instance(str)
@describe.instance(float)
def _describe_basics(instance: _BasicType) -> _BasicType:
    return instance


@describe.instance(protocol=HasDescribe)
def _describe_has_describe(instance: HasDescribe) -> JSONDataType:
    return instance.__describe__()


class ListOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, list) and all(describe.supports(item) for item in instance)


class ListOfDescribable(List[Supports[Describable]], metaclass=ListOfDescribableMeta):
    ...


@describe.instance(delegate=ListOfDescribable)
def _describe_list(instance: ListOfDescribable) -> List[JSONDataType]:
    return [describe(item) for item in instance]


class DictOfDescribableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and describe.supports(value) for key, value in instance.items()
        )


class DictOfDescribable(Dict[str, Supports[Describable]], metaclass=DictOfDescribableMeta):
    ...


@describe.instance(delegate=DictOfDescribable)
def _describe_dict(instance: DictOfDescribable) -> Dict[str, JSONDataType]:
    return {key: describe(value) for key, value in instance.items()}
