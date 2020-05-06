import enum

from sklearn.model_selection import train_test_split

class SplitSubset(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()

    @classmethod
    def from_string(cls, s, raise_if_not_found=False):
        for k, v in SplitSubset.conversion_table.items():
            if s in v:
                return k
        if raise_if_not_found:
            raise ValueError(f'string {s} was not found in the conversion table. Available values are {list(SplitSubset.conversion_table.values())}.')
        return None

SplitSubset.conversion_table = {
    SplitSubset.TRAIN: {'train'},
    SplitSubset.VAL: {'val', 'validation'},
    SplitSubset.TEST: {'test'},
}

def train_val_test_split(dataset, n_samples, train_size=None, val_size=None, test_size=None, **options):
    # TODO: currently, this function only accepts one dataset. Allow more.
    # TODO: currently, this function requires the argument n_samples. Remove this.
    # TODO: to keep consistency, allow elements from SplitSubset.conversion_table
    
    if val_size is None or val_size == 0:
        train_set, test_set = train_test_split(dataset, train_size=train_size, test_size=test_size, **options)
        val_set = []
    else:
        if 0 < val_size < 1:
            val_size = int(val_size * n_samples)
        elif val_size < 0 or val_size > n_samples:
            raise ValueError(f'invalid val_size {val_size}. Expected a float in (0, 1), or a float in [0, n_samples={n_samples}].')
        
        if train_size is None:
            train_set, test_set = train_test_split(dataset, test_size=test_size, **options)
            train_set, val_set = train_test_split(train_set, test_size=val_size, **options)
        else:
            train_set, test_set = train_test_split(dataset, train_size=train_size, **options)
            val_set, test_set = train_test_split(test_set, train_size=val_size, **options)
            
    return train_set, val_set, test_set
    
