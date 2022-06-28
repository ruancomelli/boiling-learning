from typing import Any, Tuple, TypeVar

_Any = TypeVar('_Any')
Pair = Tuple[_Any, _Any]
Triplet = Tuple[_Any, _Any, _Any]


def typename(obj: Any) -> str:
    return type(obj).__name__
