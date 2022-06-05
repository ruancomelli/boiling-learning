import math
from fractions import Fraction
from functools import reduce
from typing import Callable, Tuple

import funcy


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


def round_to_multiple(number: int, base: int, rounder: Callable[[float], int] = round) -> int:
    return base * rounder(number / base)
