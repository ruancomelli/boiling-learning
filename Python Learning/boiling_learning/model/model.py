from pathlib import Path
import enum

import parse
from sklearn.model_selection import train_test_split

import boiling_learning as bl

_sentinel = object()


class SplitSubset(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TRAIN_VAL = enum.auto()
    TEST = enum.auto()
    ALL = enum.auto()

    @classmethod
    def get_split(cls, s, default=_sentinel):
        if s in cls:
            return s
        else:
            return cls.from_string(s, default=default)

    @classmethod
    def from_string(cls, s, default=_sentinel):
        for k, v in cls.FROM_STR.items():
            if s in v:
                return k
        if default is _sentinel:
            raise ValueError(
                f'string {s} was not found in the conversion table.'
                f'Available values are {list(cls.FROM_STR.values())}.')
        else:
            return default

    def to_str(self):
        return self.name.lower()


SplitSubset.FROM_STR = {
    SplitSubset.TRAIN: {'train'},
    SplitSubset.VAL: {'val', 'validation'},
    SplitSubset.TRAIN_VAL: set(
        connector.join(['train', validation_key])
        for connector in ['_', '_and_']
        for validation_key in ['val', 'validation']
    ),
    SplitSubset.TEST: {'test'},
    SplitSubset.ALL: {'all'},
}


def train_val_test_split(
        dataset,
        n_samples,
        train_size=None,
        val_size=None,
        test_size=None,
        **options
):
    # TODO: this function only accepts one dataset. Allow more.
    # TODO: this function requires the argument n_samples. Remove this.
    # TODO: to keep consistency, allow elements from SplitSubset.FROM_STR

    if val_size is None or val_size == 0:
        train_set, test_set = train_test_split(
            dataset,
            train_size=train_size,
            test_size=test_size,
            **options
        )
        val_set = []
    else:
        if 0 < val_size < 1:
            val_size = int(val_size * n_samples)
        elif val_size < 0 or val_size > n_samples:
            raise ValueError(
                f'invalid val_size {val_size}.'
                'Expected a float in (0, 1),'
                f'or a float in [0, n_samples={n_samples}].')

        if train_size is None:
            train_set, test_set = train_test_split(
                dataset,
                test_size=test_size,
                **options
            )
            train_set, val_set = train_test_split(
                train_set,
                test_size=val_size,
                **options
            )
        else:
            train_set, test_set = train_test_split(
                dataset,
                train_size=train_size,
                **options
            )
            val_set, test_set = train_test_split(
                test_set,
                train_size=val_size,
                **options
            )

    return train_set, val_set, test_set


def restore(
    restore=False,
    path=None,
    load_method=None,
    epoch_str='epoch'
):
    last_epoch = -1
    model = None
    if restore:
        path = Path(path)
        glob_pattern = path.name.replace(f'{{{epoch_str}}}', '*')
        parser = parse.compile(path.name).parse

        paths = path.parent.glob(glob_pattern)
        parsed = (parser(path_item.name) for path_item in paths)
        parsed = filter(lambda p: p is not None and epoch_str in p, parsed)
        epochs = bl.utils.append(
            (int(p[epoch_str]) for p in parsed),
            last_epoch
        )
        last_epoch = max(epochs)

        if last_epoch != -1:
            path_str = str(path).format(epoch=last_epoch)
            model = load_method(path_str)

    return last_epoch, model
