from unittest import TestCase

from boiling_learning.io import json
from boiling_learning.io.json import JSONDataType


@json.encode.instance(tuple)
def _json_encode(obj: tuple) -> list:
    return list(obj)


@json.decode.dispatch(tuple)
def _json_decode(obj: JSONDataType) -> tuple:
    return tuple(obj)


class storage_Test(TestCase):
    def test_default_json_io(self):
        test_list = [
            314159,
            'apple pie tastes good',
            {'likes dinos': True, 'political opinion': None},
        ]

        encoded = json.dumps(json.serialize(test_list))
        decoded = json.deserialize(json.loads(encoded))

        self.assertListEqual(decoded, test_list)

    def test_custom_json_io(self):
        test_tuple = (
            314159,
            'apple pie tastes good',
            {'likes dinos': True, 'political opinion': None},
        )

        encoded = json.dumps(json.serialize(test_tuple))
        decoded = json.deserialize(json.loads(encoded))

        self.assertTupleEqual(decoded, test_tuple)


def test_json_encode_basic_types() -> None:
    assert json.encode(None) is None
    assert json.encode(3) == 3
    assert json.encode(3.14) == 3.14
    assert json.encode('hello') == 'hello'
    assert json.encode(True) is True


def test_json_encode_compound_types() -> None:
    assert json.deserialize(json.serialize([0, 1, 'hello'])) == [0, 1, 'hello']
