from fractions import Fraction
from functools import reduce
import math
from typing import (
    Tuple
)

import funcy


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
