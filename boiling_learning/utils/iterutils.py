import itertools
import typing
from collections.abc import Iterable, Iterator
from typing import TypeVar

import more_itertools as mit

_T = TypeVar("_T")


def unsort(iterable: Iterable[_T]) -> tuple[Iterable[int], Iterable[_T]]:
    peekable = mit.peekable(iterable)

    if not peekable:
        return (), ()

    sorted_items, sorters = mit.sort_together((peekable, itertools.count()))
    _, unsorters = mit.sort_together((sorters, itertools.count()))
    return (
        typing.cast(Iterable[int], unsorters),
        typing.cast(Iterable[_T], sorted_items),
    )


def accumulate_parts(iterable: Iterable[_T]) -> Iterator[tuple[_T, ...]]:
    return itertools.accumulate(
        iterable,
        lambda current, item: current + (item,),
        initial=(),
    )
