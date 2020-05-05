import enum

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
