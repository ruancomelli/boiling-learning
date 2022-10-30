from __future__ import annotations

import abc
import json as _json
from datetime import timedelta
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TypeVar, Union

from classes import AssociatedType, Supports, typeclass
from typing_extensions import Protocol, TypedDict, runtime_checkable

from boiling_learning.utils.dataclasses import is_dataclass_instance, shallow_asdict
from boiling_learning.utils.frozendicts import frozendict
from boiling_learning.utils.functional import P, Pack
from boiling_learning.utils.pathutils import PathLike, resolve
from boiling_learning.utils.table_dispatch import TableDispatcher

# see <https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support>

BasicTypes = Union[None, bool, int, str, float]
_BasicType = TypeVar('_BasicType', bound=BasicTypes)
JSONDataType = Union[BasicTypes, List['JSONDataType'], Dict[str, 'JSONDataType']]


class SerializedJSONObject(TypedDict):
    type: Optional[str]
    contents: JSONDataType


SerializedJSONDataType = Union[BasicTypes, List[BasicTypes], SerializedJSONObject]


class JSONEncodable(AssociatedType):
    ...


class JSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return encode.supports(instance)


class SupportsJSONEncodable(Supports[JSONEncodable], metaclass=JSONEncodableMeta):
    ...


@typeclass(JSONEncodable)
def encode(instance: Supports[JSONEncodable]) -> JSONDataType:
    '''Return a JSON encoding of an object.'''


@typeclass
def encode_type(instance: Any) -> JSONDataType:
    '''Return a JSON encoding of the type of an object.'''


@encode_type.instance(object)
def _encode_type_any(instance: object) -> str:
    '''Return a JSON encoding of the type of an object.'''
    return encode(type(instance))


@encode_type.instance(FunctionType)
def _encode_type_function_type(obj: FunctionType) -> str:
    return 'types.FunctionType'


decode = TableDispatcher()


@encode.instance(None)
@encode.instance(bool)
@encode.instance(int)
@encode.instance(float)
@encode.instance(str)
def _encode_basics(instance: _BasicType) -> _BasicType:
    return instance


@decode.dispatch(None)
@decode.dispatch(bool)
@decode.dispatch(int)
@decode.dispatch(float)
@decode.dispatch(str)
def _decode(obj: _BasicType) -> _BasicType:
    return obj


@runtime_checkable
class HasJSONEncode(Protocol):
    def __json_encode__(self) -> JSONDataType:
        ...


@encode.instance(protocol=HasJSONEncode)
def _encode_has_encode(instance: HasJSONEncode) -> JSONDataType:
    return instance.__json_encode__()


@encode.instance(Path)
def _encode_Path(instance: Path) -> str:
    return str(instance)


@decode.dispatch(Path)
def _decode_Path(obj: str) -> Path:
    return Path(obj)


class _PackOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Pack) and encode.supports(instance.pair())


class PackOfJSONEncodable(
    Pack[Supports[JSONEncodable], Supports[JSONEncodable]],
    metaclass=_PackOfJSONEncodableMeta,
):
    ...


@encode.instance(delegate=PackOfJSONEncodable)
def _encode_Pack(instance: PackOfJSONEncodable) -> List[SerializedJSONObject]:
    return serialize(instance.pair())


@decode.dispatch(P)
@decode.dispatch(Pack)
def _decode_Pack(obj: List[SerializedJSONObject]) -> Pack:
    args, kwargs = deserialize(obj)
    return Pack(args, kwargs)


class _ListOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, list) and all(encode.supports(item) for item in instance)


class ListOfJSONEncodable(List[Supports[JSONEncodable]], metaclass=_ListOfJSONEncodableMeta):
    ...


@encode.instance(delegate=ListOfJSONEncodable)
def _encode_list(instance: ListOfJSONEncodable) -> List[JSONDataType]:
    return [serialize(item) for item in instance]


@decode.dispatch(list)
def _decode_list(obj: List[JSONDataType]) -> list:
    return [deserialize(item) for item in obj]


class _DictOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and encode.supports(value) for key, value in instance.items()
        )


class DictOfJSONEncodable(Dict[str, Supports[JSONEncodable]], metaclass=_DictOfJSONEncodableMeta):
    ...


@encode.instance(delegate=DictOfJSONEncodable)
def _encode_dict(instance: DictOfJSONEncodable) -> Dict[str, SerializedJSONObject]:
    return {key: serialize(value) for key, value in instance.items()}


@decode.dispatch(dict)
def _decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


class _TupleOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(encode.supports(item) for item in instance)


class TupleOfJSONEncodable(
    Tuple[Supports[JSONEncodable], ...], metaclass=_TupleOfJSONEncodableMeta
):
    ...


@encode.instance(delegate=TupleOfJSONEncodable)
def _encode_tuple(instance: TupleOfJSONEncodable) -> List[SerializedJSONObject]:
    return list(map(serialize, instance))


@decode.dispatch(tuple)
def _decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


class _SetOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, set) and all(encode.supports(item) for item in instance)


class SetOfJSONEncodable(Set[Supports[JSONEncodable]], metaclass=_SetOfJSONEncodableMeta):
    ...


@encode.instance(delegate=SetOfJSONEncodable)
def _encode_set(instance: SetOfJSONEncodable) -> List[SerializedJSONObject]:
    return sorted(map(serialize, instance))


@decode.dispatch(set)
def _decode_set(obj: List[JSONDataType]) -> set:
    return set(map(deserialize, obj))


class _FrozenSetOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, frozenset) and all(encode.supports(item) for item in instance)


class FrozenSetOfJSONEncodable(
    FrozenSet[Supports[JSONEncodable]], metaclass=_FrozenSetOfJSONEncodableMeta
):
    ...


@encode.instance(delegate=FrozenSetOfJSONEncodable)
def _encode_frozenset(instance: FrozenSetOfJSONEncodable) -> List[SerializedJSONObject]:
    return sorted(map(serialize, instance))


@decode.dispatch(frozenset)
def _decode_frozenset(obj: List[JSONDataType]) -> frozenset:
    return frozenset(map(deserialize, obj))


class _FrozenDictOfJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, frozendict) and encode.supports(dict(instance))


class FrozenDictMeta(_FrozenDictOfJSONEncodableMeta, type(frozendict)):
    pass


class FrozenDictOfJSONEncodable(
    frozendict[str, Supports[JSONEncodable]],
    metaclass=FrozenDictMeta,
):
    ...


@encode.instance(delegate=FrozenDictOfJSONEncodable)
def _encode_frozendict(
    instance: FrozenDictOfJSONEncodable,
) -> Dict[str, SerializedJSONObject]:
    return encode(dict(instance))


@decode.dispatch(frozendict)
def _decode_frozendict(obj: Dict[str, Any]) -> frozendict:
    return frozendict(deserialize(obj))


class DataclassOfJSONEncodableFieldsMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return is_dataclass_instance(instance) and encode.supports(shallow_asdict(instance))


class DataclassOfJSONEncodableFields(
    metaclass=DataclassOfJSONEncodableFieldsMeta,
):
    ...


@encode.instance(delegate=DataclassOfJSONEncodableFields)
def _encode_dataclass(
    instance: DataclassOfJSONEncodableFields,
) -> Dict[str, SerializedJSONObject]:
    return serialize(shallow_asdict(instance))


@encode.instance(Fraction)
def _encode_fraction(instance: Fraction) -> List[int]:
    return serialize((instance.numerator, instance.denominator))


@decode.dispatch(Fraction)
def _decode_fraction(instance: List[int]) -> Fraction:
    numerator, denominator = instance
    return Fraction(deserialize(numerator), deserialize(denominator))


@encode.instance(timedelta)
def _encode_timedelta(instance: timedelta) -> float:
    return serialize(instance.total_seconds())


@decode.dispatch(timedelta)
def _decode_timedelta(instance: float) -> timedelta:
    return timedelta(seconds=instance)


@encode.instance(FunctionType)
@encode.instance(type)
def _encode_types(instance: type) -> str:
    return f'{instance.__module__}.{instance.__qualname__}'


@decode.dispatch(abc.ABCMeta)
@decode.dispatch(FunctionType)
@decode.dispatch(type)
def _decode_types(instance: str) -> type:
    modulepath, typename = instance.rsplit('.', maxsplit=1)
    module = import_module(modulepath)
    return getattr(module, typename)


def serialize(obj: Supports[JSONEncodable]) -> SerializedJSONObject:
    '''Return a JSON serialization of an object.'''
    if obj is None or isinstance(obj, (bool, int, str, float)):
        return obj

    if isinstance(obj, (list, tuple)):
        return _nested_sort_dicts([encode_type(obj), *(serialize(item) for item in obj)])

    return _nested_sort_dicts([encode_type(obj), encode(obj)])


def deserialize(obj: SerializedJSONDataType) -> Any:
    if isinstance(obj, dict):
        return {key: deserialize(value) for key, value in obj.items()}

    if isinstance(obj, list):
        encoded_type, *encoded_items = obj

        obj_type = decode[type](encoded_type)

        if obj_type is list:
            return [deserialize(item) for item in encoded_items]

        if obj_type is tuple:
            return tuple(deserialize(item) for item in encoded_items)

        (encoded_item,) = encoded_items
        return decode[obj_type](encoded_item)

    return obj


def dumps(obj: Supports[JSONEncodable]) -> str:
    return _json.dumps(serialize(obj))


def dump(obj: Supports[JSONEncodable], path: PathLike) -> None:
    serialized = serialize(obj)

    with resolve(path, parents=True).open('w', encoding='utf-8') as file:
        _json.dump(serialized, file, indent=4)


def load(path: PathLike) -> Any:
    with resolve(path).open('r', encoding='utf-8') as file:
        obj = _json.load(file)

    return deserialize(obj)


def loads(contents: str) -> Any:
    return deserialize(_json.loads(contents))


def _nested_sort_dicts(obj: SerializedJSONObject) -> SerializedJSONObject:
    return (
        {k: _nested_sort_dicts(v) for k, v in sorted(obj.items())}
        if isinstance(obj, dict)
        else [_nested_sort_dicts(item) for item in obj]
        if isinstance(obj, list)
        else obj
    )
