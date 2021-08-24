from importlib import import_module
from typing import Any, Dict, List, TypeVar

from plum import Dispatcher

from boiling_learning.utils.functional import Pack
from boiling_learning.utils.table_dispatch import table_dispatch
from boiling_learning.utils.utils import JSONDataType

_T = TypeVar('_T')
dispatch = Dispatcher()


@dispatch
def json_encode(obj: None) -> None:
    # default implementation: return object unmodified
    return obj


@json_encode.dispatch
def json_encode(obj: bool) -> bool:
    return obj


@json_encode.dispatch
def json_encode(obj: int) -> int:
    return obj


@json_encode.dispatch
def json_encode(obj: float) -> float:
    return obj


@json_encode.dispatch
def json_encode(obj: str) -> str:
    return obj


@json_encode.dispatch
def json_encode(obj: list) -> list:
    return list(map(json_serialize, obj))


@json_encode.dispatch
def json_encode(obj: tuple) -> list:
    return list(map(json_serialize, obj))


@json_encode.dispatch
def json_encode(obj: dict) -> dict:
    return {key: json_serialize(value) for key, value in obj.items()}


@json_encode.dispatch
def json_encode(obj: Pack) -> list:
    return [json_serialize(obj.args), json_serialize(dict(obj.kwargs))]


json_decode = table_dispatch()


@json_decode.dispatch(None)
@json_decode.dispatch(bool)
@json_decode.dispatch(int)
@json_decode.dispatch(float)
@json_decode.dispatch(str)
def _json_decode(obj: _T) -> _T:
    return obj


@json_decode.dispatch(list)
def _json_decode_list(obj: List[JSONDataType]) -> list:
    return list(map(json_deserialize, obj))


@json_decode.dispatch(tuple)
def _json_decode_tuple(obj: List[JSONDataType]) -> tuple:
    return tuple(map(json_deserialize, obj))


@json_decode.dispatch(dict)
def _json_decode_dict(obj: Dict[str, JSONDataType]) -> dict:
    return {key: json_deserialize(value) for key, value in obj.items()}


@json_decode.dispatch(Pack)
def _json_decode_Pack(obj: JSONDataType) -> Pack:
    args, kwargs = obj
    return Pack(json_deserialize(args), json_deserialize(kwargs))


@dispatch
def json_serialize(obj: Any) -> Dict[str, Any]:
    return {
        'type': f'{type(obj).__module__}.{type(obj).__qualname__}',
        'contents': json_encode(obj),
    }


@json_serialize.dispatch
def json_serialize(obj: None) -> Dict[str, Any]:
    return {'type': None, 'contents': json_encode(obj)}


def json_deserialize(obj: Dict[str, Any]) -> Any:
    obj_type = obj['type']

    if obj_type is not None:
        modulepath, typename = obj_type.rsplit('.', maxsplit=1)
        module = import_module(modulepath)
        obj_type = getattr(module, typename)

    return json_decode[obj_type](obj['contents'])
