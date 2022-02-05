from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, TypeVar, final

from classes import AssociatedType, Supports, typeclass

from boiling_learning.io import json
from boiling_learning.utils.table_dispatch import TableDispatcher
from boiling_learning.utils.utils import JSONDataType, PathLike, resolve

# pylint: disable=missing-function-docstring,missing-class-docstring

_T = TypeVar('_T')
Metadata = JSONDataType


@final
class Serializable(AssociatedType):
    ...


class _SerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return serialize.supports(instance)


class SupportsSerializable(Supports[Serializable], metaclass=_SerializableMeta):
    ...


@typeclass(Serializable)
def serialize(instance: Supports[Serializable], path: Path) -> Metadata:
    '''Serialize object contents.'''


class _DeserializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return instance in deserialize


class SupportsDeserializable(metaclass=_DeserializableMeta):
    ...


deserialize = TableDispatcher()


class Saveable(AssociatedType):
    ...


class _SupportsSaveableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
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

    serialization_metadata = serialize(instance, path / '__data__')

    metadata_path = path / '__boiling_learning_save_meta__.json'
    metadata = {
        'type': type(instance) if instance is not None else None,
        'metadata': serialization_metadata,
    }
    json.dump(metadata, metadata_path)


def load(path: PathLike) -> Any:
    path = resolve(path)

    metadata_path = path / '__boiling_learning_save_meta__.json'
    metadata = json.load(metadata_path)

    obj_type = metadata['type']

    return deserialize[obj_type](path / '__data__', metadata=metadata['metadata'])


class _SimpleJSONSerializableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return instance is None or (
            isinstance(instance, (bool, int, float, str, dict, list, tuple))
            and json.serialize.supports(instance)
        )


class SimpleJSONSerializable(metaclass=_SimpleJSONSerializableMeta):
    pass


@serialize.instance(delegate=SimpleJSONSerializable)
def _serialize_simple_json_serializable(instance: SimpleJSONSerializable, path: Path) -> Metadata:
    json.dump(instance, path)
    return {'json': True}


class _ListOfSaveableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, list) and all(save.supports(item) for item in instance)


class ListOfSaveable(metaclass=_ListOfSaveableMeta):
    pass


@serialize.instance(delegate=ListOfSaveable)
def _serialize_list(instance: ListOfSaveable, path: Path) -> None:
    path = resolve(path, dir=True)

    for index, item in enumerate(instance):
        save(item, path / str(index))


class _DictOfSaveableMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, dict) and all(
            isinstance(key, str) and save.supports(value) for key, value in instance.items()
        )


class DictOfSaveable(metaclass=_DictOfSaveableMeta):
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
    return json.load(path)


@deserialize.dispatch(bool)
def _deserialize_bool(path: Path, metadata: Metadata) -> bool:  # pylint: disable=unused-argument
    return json.load(path)


@deserialize.dispatch(float)
def _deserialize_float(path: Path, metadata: Metadata) -> float:  # pylint: disable=unused-argument
    return json.load(path)


@deserialize.dispatch(str)
def _deserialize_str(path: Path, metadata: Metadata) -> str:  # pylint: disable=unused-argument
    return json.load(path)


@deserialize.dispatch(list)
def _deserialize_list(path: Path, metadata: Metadata) -> list:
    if metadata.get('json'):
        return json.load(path)

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
    if metadata.get('json'):
        return json.load(path)

    return {item_path.name: load(item_path) for item_path in path.iterdir()}
