from boiling_learning.describe.described import Described
from boiling_learning.describe.describers import describe


def test_described_list() -> None:
    described = Described.from_list([])

    assert described() == []
    assert describe(described) == []
