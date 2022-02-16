from boiling_learning.utils.described import Described
from boiling_learning.utils.descriptions import describe


def test_described_list() -> None:
    described = Described.from_list([])

    assert described.value == []
    assert describe(described) == []
