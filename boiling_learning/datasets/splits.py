from fractions import Fraction
from typing import Generic, TypeVar

import funcy

# generic `NamedTuple`s were only introduced in Python 3.11 - until then we need to
# import from `typing_extensions`
from typing_extensions import NamedTuple

from boiling_learning.io.dataclasses import dataclass

_T = TypeVar('_T')


class DatasetTriplet(NamedTuple, Generic[_T]):
    train: _T
    val: _T
    test: _T


@dataclass(frozen=True)
class DatasetSplits:
    train: Fraction | None = None
    test: Fraction | None = None
    val: Fraction | None = Fraction(0)

    def __post_init__(self) -> None:
        splits = (self.train, self.val, self.test)
        n_nones = splits.count(None)
        if n_nones > 1:
            raise ValueError(
                'at most one of *train*, *val* and *test* can be inferred (by passing `None`)'
            )

        if n_nones == 1:
            names = ('train', 'val', 'test')
            dct = funcy.zipdict(names, splits)
            for name, split in dct.items():
                if split is None:
                    others = funcy.omit(dct, [name])
                    others_sum = sum(others.values())

                    if not 0 < others_sum <= 1:
                        raise ValueError(
                            'it is required that 0 < '
                            + ' + '.join(f'*{other}*' for other in others.keys())
                            + ' <= 1'
                        )

                    split = 1 - others_sum
                    object.__setattr__(self, name, split)
                    splits = (self.train, self.val, self.test)
                    break

        if sum(splits) != 1:
            raise ValueError('*train* + *val* + *test* must equal 1')

        if not (0 < self.train < 1 and 0 <= self.val < 1 and 0 < self.test < 1):
            raise ValueError('it is required that 0 < (*train*, *test*) < 1 and 0 <= *val* < 1')
