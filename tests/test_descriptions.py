from boiling_learning.descriptions import describe
from boiling_learning.utils.functional import P


class TestDescribeDefault:
    def test_pack(self) -> None:
        assert describe(P()) == ((), {})
