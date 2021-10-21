from __future__ import annotations

import enum
from typing import TypeVar, Union

_T = TypeVar('_T')


class Sentinel(enum.Enum):
    INSTANCE = enum.auto()


EMPTY = Sentinel.INSTANCE
Emptiable = Union[Sentinel, _T]
