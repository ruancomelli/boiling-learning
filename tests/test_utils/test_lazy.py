from boiling_learning.descriptions import describe
from boiling_learning.utils.lazy import LazyDescribed


def test_lazy_described_from_list() -> None:
    described = LazyDescribed.from_list([])

    assert described() == []
    assert describe(described) == []
