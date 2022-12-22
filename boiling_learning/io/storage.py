from __future__ import annotations

import itertools
from dataclasses import dataclass as _dataclass
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Type, TypeVar, final

from classes import AssociatedType, Supports, typeclass

from boiling_learning.dataclasses import (
    DataClass,
    is_dataclass_class,
    is_dataclass_instance,
    shallow_asdict,
)
from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.lazy import LazyDescribed
from boiling_learning.utils.pathutils import PathLike, resolve
from boiling_learning.utils.table_dispatch import TableDispatcher

Metadata = json.JSONDataType
_DataClassType = TypeVar('_DataClassType', bound=Type[DataClass])
_AnyCallable = TypeVar('_AnyCallable', bound=Callable[..., Any])

DATA_FILENAME = '__data__'
METADATA_FILENAME = '__boiling_learning_save_meta__.json'


@final
class Serializable(AssociatedType):
    ...


class _SerializableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return serialize.supports(instance)


class SupportsSerializable(Supports[Serializable], metaclass=_SerializableMeta):
    ...


@typeclass(Serializable)
def serialize(instance: Supports[Serializable], path: Path) -> Metadata:
    '''Serialize object contents.'''


class _DeserializableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return instance in deserialize


class SupportsDeserializable(metaclass=_DeserializableMeta):
    ...


deserialize = TableDispatcher()


class Saveable(AssociatedType):
    ...


class _SupportsSaveableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return save.supports(instance)


class SupportsSaveable(
    Supports[Saveable],
    metaclass=_SupportsSaveableMeta,
):
    ...


@typeclass(Saveable)
def save(obj: Supports[Saveable], path: PathLike) -> None:
    '''Save objects.'''


@save.instance(delegate=SupportsSerializable)
def _save_serializable(instance: Supports[Serializable], path: PathLike) -> None:
    path = resolve(path, dir=True)

    serialization_metadata = serialize(instance, path / DATA_FILENAME)

    metadata = {
        'type': type(instance) if instance is not None else None,
        'metadata': serialization_metadata,
    }
    json.dump(metadata, path / METADATA_FILENAME)


def load(path: PathLike) -> Any:
    path = resolve(path)

    metadata = json.load(path / METADATA_FILENAME)

    obj_type = metadata['type']

    return deserialize[obj_type](path / DATA_FILENAME, metadata['metadata'])


class _SimpleJSONEncodableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return instance is None or (
            isinstance(instance, (bool, int, float, str, dict, list, tuple))
            and json.encode.supports(instance)
        )


class SimpleJSONEncodable(metaclass=_SimpleJSONEncodableMeta):
    pass


@serialize.instance(delegate=SimpleJSONEncodable)
def _serialize_simple_json_serializable(instance: SimpleJSONEncodable, path: Path) -> Metadata:
    json.dump(instance, path.with_suffix('.json'))
    return {'json': True}


class _ListOfSaveableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, list) and all(save.supports(item) for item in instance)


class ListOfSaveable(list[Supports[Saveable]], metaclass=_ListOfSaveableMeta):
    pass


@serialize.instance(delegate=ListOfSaveable)
def _serialize_list(instance: ListOfSaveable, path: Path) -> None:
    path = resolve(path, dir=True)

    for index, item in enumerate(instance):
        save(item, path / str(index))


class _TupleOfSaveableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, tuple) and all(save.supports(item) for item in instance)


class TupleOfSaveable(tuple[Supports[Saveable], ...], metaclass=_TupleOfSaveableMeta):
    pass


@serialize.instance(delegate=TupleOfSaveable)
def _serialize_tuple(instance: TupleOfSaveable, path: Path) -> None:
    serialize(list(instance), path)


class _DictOfSaveableMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and save.supports(value) for key, value in instance.items()
        )


class DictOfSaveable(dict[str, Supports[Saveable]], metaclass=_DictOfSaveableMeta):
    pass


@serialize.instance(delegate=DictOfSaveable)
def _serialize_dict(instance: DictOfSaveable, path: Path) -> None:
    path = resolve(path, dir=True)

    for key, value in instance.items():
        save(value, path / str(key))


@deserialize.dispatch(None)
def _deserialize_none(path: Path, metadata: Metadata) -> None:  # pylint: disable=unused-argument
    return None


@deserialize.dispatch(int)
def _deserialize_int(path: Path, metadata: Metadata) -> int:  # pylint: disable=unused-argument
    return json.load(path.with_suffix('.json'))


@deserialize.dispatch(bool)
def _deserialize_bool(path: Path, metadata: Metadata) -> bool:  # pylint: disable=unused-argument
    return json.load(path.with_suffix('.json'))


@deserialize.dispatch(float)
def _deserialize_float(path: Path, metadata: Metadata) -> float:  # pylint: disable=unused-argument
    return json.load(path.with_suffix('.json'))


@deserialize.dispatch(str)
def _deserialize_str(path: Path, metadata: Metadata) -> str:  # pylint: disable=unused-argument
    return json.load(path.with_suffix('.json'))


@deserialize.dispatch(list)
def _deserialize_list(path: Path, metadata: Metadata) -> list:
    if isinstance(metadata, dict) and metadata.get('json'):
        return json.load(path.with_suffix('.json'))

    items = []

    for index in itertools.count():
        item_path = path / str(index)

        if item_path.exists():
            items.append(load(item_path))
        else:
            break

    return items


@deserialize.dispatch(dict)
def _deserialize_dict(path: Path, metadata: Metadata) -> dict:
    if isinstance(metadata, dict) and metadata.get('json'):
        return json.load(path.with_suffix('.json'))

    return {item_path.name: load(item_path) for item_path in path.iterdir()}


@deserialize.dispatch(tuple)
def _deserialize_tuple(path: Path, metadata: Metadata) -> tuple:
    return tuple(deserialize[list](path, metadata))


@serialize.instance(timedelta)
def _serialize_timedelta(instance: timedelta, path: Path) -> None:
    save(instance.total_seconds(), path)


@deserialize.dispatch(timedelta)
def _deserialize_timedelta(path: Path, _metadata: Metadata) -> timedelta:
    return timedelta(seconds=load(path))


@serialize.instance(LazyDescribed)
def _serialize_lazy_described(instance: LazyDescribed[Any], path: Path) -> None:
    save(instance(), path / 'value')
    save(describe(instance), path / 'description')


@deserialize.dispatch(LazyDescribed)
def _deserialize_lazy_described(path: Path, _metadata: Metadata) -> LazyDescribed:
    return LazyDescribed.from_value_and_description(
        load(path / 'value'),
        load(path / 'description'),
    )


class DataclassOfSaveableFieldsMeta(type):
    def __instancecheck__(self, instance: Any) -> bool:
        return is_dataclass_instance(instance) and save.supports(shallow_asdict(instance))


class DataclassOfSaveableFields(
    DataClass,
    metaclass=DataclassOfSaveableFieldsMeta,
):
    ...


@serialize.instance(delegate=DataclassOfSaveableFields)
def _serialize_dataclass(instance: DataclassOfSaveableFields, path: Path) -> Metadata:
    return serialize(shallow_asdict(instance), path)


def register_deserializer_for_dataclass(dataclass_type: _DataClassType) -> _DataClassType:
    @deserialize.dispatch(dataclass_type)
    def _deserialize(path: Path, metadata: Metadata) -> DataClass:
        fields = deserialize[dict](path, metadata)
        return dataclass_type(**fields)

    return dataclass_type


def _auto_register_deserializer_for_dataclass(dataclass_decorator: _AnyCallable) -> _AnyCallable:
    @wraps(dataclass_decorator)
    def _wrapped(*args, **kwargs) -> Any:
        decorated_result = dataclass_decorator(*args, **kwargs)

        if is_dataclass_class(decorated_result):
            # This is the case where we are directly wrapping a class, like
            #
            # @dataclass
            # class C: pass
            #
            # In this situation, directly register the class.
            return register_deserializer_for_dataclass(decorated_result)
        else:
            # This is the case where we are first passing arguments to `dataclass`
            # before actually decorating a class, like
            #
            # @dataclass(frozen=True)
            # class C: pass
            #
            # In this situation, recurse by instructing the new decorator to
            # automatically register dataclasses.
            return _auto_register_deserializer_for_dataclass(decorated_result)

    return _wrapped


dataclass = _auto_register_deserializer_for_dataclass(_dataclass)
