from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, Optional, Tuple, TypeVar

import funcy

from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.utils import resolve

_T = TypeVar('_T')


class DatasetTriplet(Tuple[_T, _T, _T], Generic[_T]):
    pass


@dataclass(frozen=True)
class DatasetSplits:
    train: Optional[Fraction] = None
    test: Optional[Fraction] = None
    val: Optional[Fraction] = Fraction(0)

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


@serialize.instance(DatasetTriplet)
def _serialize_dataset_triplet(instance: DatasetTriplet[Any], path: Path) -> None:
    path = resolve(path, dir=True)

    ds_train, ds_val, ds_test = instance

    save(ds_train, path / 'train')
    save(ds_val, path / 'val')
    save(ds_test, path / 'test')


@deserialize.dispatch(DatasetTriplet)
def _deserialize_dataset_triplet(path: Path, metadata: Metadata) -> DatasetTriplet[Any]:
    ds_train = load(path / 'train')
    ds_val = load(path / 'val')
    ds_test = load(path / 'test')

    return DatasetTriplet(ds_train, ds_val, ds_test)
