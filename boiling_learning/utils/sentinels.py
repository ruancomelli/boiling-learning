from __future__ import annotations

import enum
from typing import TypeVar, Union

_T = TypeVar('_T')


class _Sentinel(enum.Enum):
    INSTANCE = enum.auto()


EMPTY = _Sentinel.INSTANCE
Emptiable = Union[_Sentinel, _T]
