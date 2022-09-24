from boiling_learning.io import json


def _example_function(x: int, y: str) -> str:
    return y * x


class TestIdentity:
    def test_functions(self) -> None:
        print(json.serialize(_example_function))
        assert json.deserialize(json.serialize(_example_function)) is _example_function
