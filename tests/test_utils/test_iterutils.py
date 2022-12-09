from boiling_learning.utils.iterutils import accumulate_parts


class TestAccumulatedParts:
    def test_empty(self) -> None:
        assert tuple(accumulate_parts(())) == ((),)

    def test_single(self) -> None:
        assert tuple(accumulate_parts((1,))) == ((), (1,))

    def test_many(self) -> None:
        assert tuple(accumulate_parts((1, 2, 3))) == ((), (1,), (1, 2), (1, 2, 3))
