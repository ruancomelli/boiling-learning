import enum
import math
from fractions import Fraction
from functools import reduce
from typing import Iterable, Tuple, TypeVar, Union, overload

import funcy

from boiling_learning.utils.typeutils import SupportsLessThanT

_T = TypeVar('_T')
_U = TypeVar('_U')


class _SentinelType(enum.Enum):
    INSTANCE = enum.auto()


_sentinel = _SentinelType.INSTANCE


def gcd(*args: int) -> int:
    if len(args) < 2:
        raise TypeError('*gcd* requires 2 or more arguments.')

    return reduce(math.gcd, args)


def lcm(*args: int) -> int:
    if len(args) < 2:
        raise TypeError('*lcm* requires 2 or more arguments.')

    def _lcm(x: int, y: int) -> int:
        return abs(x * y) // gcd(x, y)

    return reduce(_lcm, args)


def proportional_ints(*args: Fraction) -> Tuple[int, ...]:
    denominators = funcy.pluck_attr('denominator', args)
    denominators_lcm = lcm(*denominators)

    def _proportional_int(arg: Fraction) -> int:
        return int(arg * denominators_lcm)

    ints = map(_proportional_int, args)
    return tuple(ints)


@overload
def minmax(
    iterable: Iterable[SupportsLessThanT],
) -> Tuple[SupportsLessThanT, SupportsLessThanT]:
    ...


@overload
def minmax(
    iterable: Iterable[SupportsLessThanT], default: _U
) -> Tuple[Union[SupportsLessThanT, _U], Union[SupportsLessThanT, _U]]:
    ...


def minmax(
    iterable: Iterable[SupportsLessThanT],
    default: Union[_SentinelType, _U] = _sentinel,
) -> Tuple[Union[SupportsLessThanT, _U], Union[SupportsLessThanT, _U]]:
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration as e:
        if default is _sentinel:
            raise ValueError('got an empty iterable!') from e

        return (default, default)

    lo, hi = first, first
    for val in it:
        if val < lo:
            lo = val
        elif hi < val:
            hi = val

    return lo, hi
