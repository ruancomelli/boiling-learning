import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, TypeVar

from plum import Dispatcher

from boiling_learning.utils.functional import Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType, PathLike, resolve

_T = TypeVar('_T')
dispatch = Dispatcher()


@dispatch
def encode(obj: None) -> None:
    # default implementation: return object unmodified
    return obj


@encode.dispatch
def _encode_bool(obj: bool) -> bool:
    return obj


@encode.dispatch
def _encode_int(obj: int) -> int:
    return obj


@encode.dispatch
def _encode_float(obj: float) -> float:
    return obj


@encode.dispatch
def _encode_str(obj: str) -> str:
    return obj


@encode.dispatch
def _encode_list(obj: list) -> list:
    return list(map(serialize, obj))


@encode.dispatch
def _encode_tuple(obj: tuple) -> list:
    return list(map(serialize, obj))


@encode.dispatch
def _encode_dict(obj: dict) -> dict:
    return {key: serialize(value) for key, value in obj.items()}


@encode.dispatch
def _encode_Pack(obj: Pack) -> list:
    return [serialize(obj.args), serialize(dict(obj.kwargs))]


@encode.dispatch
def _encode_Path(obj: Path) -> str:
    return str(obj)


decode = table_dispatch()


@decode.dispatch(None)
@decode.dispatch(bool)
@decode.dispatch(int)
@decode.dispatch(float)
@decode.dispatch(str)
def _json_decode(obj: _T) -> _T:
    return obj


@decode.dispatch(list)
def _json_decode_list(obj: List[JSONDataType]) -> list:
    return list(map(deserialize, obj))


@decode.dispatch(tuple)
def _json_decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(deserialize, obj))


@decode.dispatch(dict)
def _json_decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: deserialize(value) for key, value in obj.items()}


@decode.dispatch(Pack)
def _json_decode_Pack(obj: JSONDataType) -> Pack:
    args, kwargs = obj
    return Pack(deserialize(args), deserialize(kwargs))


@decode.dispatch(Path)
def _json_decode_Path(obj: str) -> Path:
    return Path(obj)


@dispatch
def serialize(obj: Any) -> Dict[str, Any]:
    return {
        'type': f'{type(obj).__module__}.{type(obj).__qualname__}',
        'contents': encode(obj),
    }


@serialize.dispatch
def _serialize_None(obj: None) -> Dict[str, Any]:
    return {'type': None, 'contents': encode(obj)}


def deserialize(obj: Dict[str, Any]) -> Any:
    obj_type = obj['type']

    if obj_type is not None:
        modulepath, typename = obj_type.rsplit('.', maxsplit=1)
        module = import_module(modulepath)
        obj_type = getattr(module, typename)

    return decode[obj_type](obj['contents'])


def save(obj: Any, path: PathLike) -> None:
    serialized = serialize(obj)

    with resolve(path, parents=True).open('w') as file:
        json.dump(serialized, file, indent=4)


def load(path: PathLike) -> Any:
    with resolve(path, parents=True).open('r') as file:
        obj = json.load(file)

    return deserialize(obj)
