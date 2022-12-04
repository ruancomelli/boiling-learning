from dataclasses import dataclass

from boiling_learning.dataclasses import is_dataclass, is_dataclass_class, is_dataclass_instance


@dataclass
class MyDataclass:
    x: int
    y: str


def test_is_dataclass() -> None:
    assert not is_dataclass({'x': 1, 'y': 'hi'})
    assert is_dataclass(MyDataclass)
    assert is_dataclass(MyDataclass(x=1, y='hi'))


def test_is_dataclass_class() -> None:
    assert not is_dataclass_class({'x': 1, 'y': 'hi'})
    assert is_dataclass_class(MyDataclass)
    assert not is_dataclass_class(MyDataclass(x=1, y='hi'))


def test_is_dataclass_instance() -> None:
    assert not is_dataclass_instance({'x': 1, 'y': 'hi'})
    assert not is_dataclass_instance(MyDataclass)
    assert is_dataclass_instance(MyDataclass(x=1, y='hi'))
