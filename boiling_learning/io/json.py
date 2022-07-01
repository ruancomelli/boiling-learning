from __future__ import annotations

import abc
import json as _json
from datetime import timedelta
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TypeVar, Union

from classes import AssociatedType, Supports
from classes import typeclass as _typeclass
from typing_extensions import Protocol, TypedDict, runtime_checkable

from boiling_learning.utils import PathLike, resolve
from boiling_learning.utils.dataclasses import is_dataclass_instance, shallow_asdict
from boiling_learning.utils.frozendict import frozendict
from boiling_learning.utils.functional import P, Pack
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


@_typeclass(JSONEncodable)
def encode(instance: Supports[JSONEncodable]) -> JSONDataType:
    '''Return a JSON encoding of an object.'''


decode = TableDispatcher()


class JSONSerializable(AssociatedType):
    ...


class SupportsJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return serialize.supports(instance)


class SupportsJSONSerializable(
    Supports[JSONSerializable],
    metaclass=SupportsJSONSerializableMeta,
):
    ...


@_typeclass(JSONSerializable)
def serialize(obj: Supports[JSONSerializable]) -> SerializedJSONObject:
    '''Return a JSON serialization of an object.'''


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


class _PackOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, Pack) and serialize.supports(instance.pair())


class PackOfJSONSerializable(
    Pack[Supports[JSONSerializable], Supports[JSONSerializable]],
    metaclass=_PackOfJSONSerializableMeta,
):
    ...


@encode.instance(delegate=PackOfJSONSerializable)
def _encode_Pack(instance: PackOfJSONSerializable) -> List[SerializedJSONObject]:
    return serialize(instance.pair())


@decode.dispatch(P)
@decode.dispatch(Pack)
def _decode_Pack(obj: List[SerializedJSONObject]) -> Pack:
    args, kwargs = deserialize(obj)
    return Pack(args, kwargs)


class _ListOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, list) and all(serialize.supports(item) for item in instance)


class ListOfJSONSerializable(
    List[Supports[JSONSerializable]], metaclass=_ListOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=ListOfJSONSerializable)
def _encode_list(instance: ListOfJSONSerializable) -> List[JSONDataType]:
    return [serialize(item) for item in instance]


@decode.dispatch(list)
def _decode_list(obj: List[JSONDataType]) -> list:
    return list(map(deserialize, obj))


class _DictOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and serialize.supports(value) for key, value in instance.items()
        )


class DictOfJSONSerializable(
    Dict[str, Supports[JSONSerializable]], metaclass=_DictOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=DictOfJSONSerializable)
def _encode_dict(instance: DictOfJSONSerializable) -> Dict[str, SerializedJSONObject]:
    return {key: serialize(value) for key, value in instance.items()}


@decode.dispatch(dict)
def _decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


class _TupleOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(serialize.supports(item) for item in instance)


class TupleOfJSONSerializable(
    Tuple[Supports[JSONSerializable], ...], metaclass=_TupleOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=TupleOfJSONSerializable)
def _encode_tuple(instance: TupleOfJSONSerializable) -> List[SerializedJSONObject]:
    return list(map(serialize, instance))


@decode.dispatch(tuple)
def _decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


class _SetOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, set) and all(serialize.supports(item) for item in instance)


class SetOfJSONSerializable(Set[Supports[JSONSerializable]], metaclass=_SetOfJSONSerializableMeta):
    ...


@encode.instance(delegate=SetOfJSONSerializable)
def _encode_set(instance: SetOfJSONSerializable) -> List[SerializedJSONObject]:
    return sorted(map(serialize, instance))


@decode.dispatch(set)
def _decode_set(obj: List[JSONDataType]) -> set:
    return set(map(deserialize, obj))


class _FrozenSetOfJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, frozenset) and all(
            serialize.supports(item) for item in instance
        )


class FrozenSetOfJSONSerializable(
    FrozenSet[Supports[JSONSerializable]], metaclass=_FrozenSetOfJSONSerializableMeta
):
    ...


@encode.instance(delegate=FrozenSetOfJSONSerializable)
def _encode_frozenset(instance: FrozenSetOfJSONSerializable) -> List[SerializedJSONObject]:
    return sorted(map(serialize, instance))


@decode.dispatch(frozenset)
def _decode_frozenset(obj: List[JSONDataType]) -> frozenset:
    return frozenset(map(deserialize, obj))


class _FrozenDictOfJSONEncodableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, frozendict) and serialize.supports(dict(instance))


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
    return serialize(dict(instance))


@decode.dispatch(frozendict)
def _decode_frozendict(obj: Dict[str, Any]) -> frozendict:
    return frozendict(deserialize(obj))


class DataclassOfJSONSerializableFieldsMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return is_dataclass_instance(instance) and serialize.supports(shallow_asdict(instance))


class DataclassOfJSONSerializableFields(
    metaclass=DataclassOfJSONSerializableFieldsMeta,
):
    ...


@encode.instance(delegate=DataclassOfJSONSerializableFields)
def _encode_dataclass(
    instance: DataclassOfJSONSerializableFields,
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


@encode.instance(type)
def _encode_types(instance: type) -> str:
    return f'{instance.__module__}.{instance.__qualname__}'


@decode.dispatch(abc.ABCMeta)
@decode.dispatch(type)
def _decode_types(instance: str) -> type:
    modulepath, typename = instance.rsplit('.', maxsplit=1)
    module = import_module(modulepath)
    return getattr(module, typename)


@serialize.instance(None)
@serialize.instance(bool)
@serialize.instance(int)
@serialize.instance(str)
@serialize.instance(float)
def _serialize_basics(instance: _BasicType) -> _BasicType:
    return instance


@serialize.instance(delegate=ListOfJSONSerializable)
def _serialize_list(instance: ListOfJSONSerializable) -> List[JSONDataType]:
    return [serialize(item) for item in instance]


class _ComplexJSONEncodableTypeMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return (
            instance is not None
            and not isinstance(instance, (bool, int, str, float, ListOfJSONSerializable))
            and encode.supports(instance)
        )


class ComplexJSONEncodableType(
    metaclass=_ComplexJSONEncodableTypeMeta,
):
    ...


@serialize.instance(delegate=ComplexJSONEncodableType)
def _serialize_complex_json_encodable(obj: ComplexJSONEncodableType) -> SerializedJSONObject:
    '''Return a JSON serialization of an object.'''
    return {
        'type': encode(type(obj)),
        'contents': encode(obj),
    }


def dumps(obj: Supports[JSONSerializable]) -> str:
    return _json.dumps(_maybe_sort_dict(serialize(obj)))


def dump(obj: Supports[JSONSerializable], path: PathLike) -> None:
    serialized: SerializedJSONObject = _maybe_sort_dict(serialize(obj))

    with resolve(path, parents=True).open('w', encoding='utf-8') as file:
        _json.dump(serialized, file, indent=4)


def deserialize(obj: SerializedJSONDataType) -> Any:
    if isinstance(obj, dict):
        encoded_type = obj['type']

        obj_type = decode[type](encoded_type) if encoded_type is not None else None

        return decode[obj_type](obj['contents'])
    elif isinstance(obj, list):
        return [deserialize(item) for item in obj]
    else:
        return obj


def load(path: PathLike) -> Any:
    with resolve(path).open('r', encoding='utf-8') as file:
        obj = _json.load(file)

    return deserialize(obj)


def loads(contents: str) -> Any:
    return deserialize(_json.loads(contents))


def _maybe_sort_dict(obj: Any) -> Any:
    return _sort_dict(obj) if isinstance(obj, dict) else obj


def _sort_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    return dict(sorted(d.items()))
