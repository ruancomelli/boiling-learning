import typing
from collections.abc import Callable, Mapping
from dataclasses import asdict, field, fields, is_dataclass
from typing import Any, Protocol, TypeGuard, TypeVar

__all__ = (
    'asdict',
    'field',
    'fields',
    'is_dataclass',
    'is_dataclass_class',
    'is_dataclass_instance',
)


class DataClass(Protocol):
    __dataclass_fields__: dict[str, Any]


_DataClass = TypeVar('_DataClass', bound=DataClass)


def is_dataclass_class(obj: Any) -> TypeGuard[type[DataClass]]:
    return isinstance(obj, type) and is_dataclass(obj)


def is_dataclass_instance(obj: Any) -> TypeGuard[DataClass]:
    return not isinstance(obj, type) and is_dataclass(obj)


def dataclass_from_mapping(
    mapping: Mapping[str, Any],
    dataclass_factory: Callable[..., _DataClass],
    key_map: Mapping[str, str] | None = None,
) -> _DataClass:
    if not is_dataclass_class(dataclass_factory):
        raise ValueError('*dataclass_factory* must be a dataclass.')

    dataclass_field_names = frozenset(field.name for field in fields(dataclass_factory))

    if key_map is not None:
        translator = {
            original_name: final_name
            for final_name, original_name in key_map.items()
            if final_name in dataclass_field_names
        }
        mapping = {translator.get(key, key): value for key, value in mapping.items()}

    return typing.cast(
        _DataClass,
        dataclass_factory(
            **{key: value for key, value in mapping.items() if key in dataclass_field_names}
        ),
    )


def shallow_asdict(obj: DataClass) -> dict[str, Any]:
    """Version of `asdict` that does not deepcopy objects.

    See https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict for this
    suggestion and its implementation.
    """
    return {field.name: getattr(obj, field.name) for field in fields(obj)}
