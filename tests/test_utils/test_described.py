from boiling_learning.describe.describers import describe
from boiling_learning.utils.described import Described


def test_described_list() -> None:
    described = Described.from_list([])

    assert described.value == []
    assert describe(described) == []
