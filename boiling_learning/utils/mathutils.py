import math
from fractions import Fraction
from functools import reduce
from typing import Any, Iterable, Tuple

import funcy

_sentinel = object()


def gcd(*args: int) -> int:
    _gcd = math.gcd

    if len(args) < 2:
        raise TypeError('*gcd* requires 2 or more arguments.')
    else:
        return reduce(_gcd, args)


def lcm(*args: int) -> int:
    def _lcm(x, y):
        return abs(x * y) // gcd(x, y)

    if len(args) < 2:
        raise TypeError('*lcm* requires 2 or more arguments.')
    else:
        return reduce(_lcm, args)


def proportional_ints(*args: Fraction) -> Tuple[int, ...]:
    denominators = funcy.pluck_attr('denominator', args)
    denominators_lcm = lcm(*denominators)

    def _proportional_int(arg: Fraction) -> int:
        return int(arg * denominators_lcm)

    ints = map(_proportional_int, args)
    return tuple(ints)


def minmax(iterable: Iterable, default: Any = _sentinel) -> Tuple[Any, Any]:
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        if default is _sentinel:
            raise ValueError('got an empty iterable!')
        else:
            return (default, default)

    lo, hi = first, first
    for val in it:
        if val < lo:
            lo = val
        elif hi < val:
            hi = val

    return lo, hi
