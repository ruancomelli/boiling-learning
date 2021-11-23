from typing import Dict
from unittest import TestCase

from boiling_learning.io import json
from boiling_learning.io.json import JSONDataType


class X:
    def __init__(self, value: int) -> None:
        self.value: int = value


@json.encode.instance(X)
def _json_encode(obj: X) -> Dict[str, int]:
    return {'value': obj.value}


@json.decode.dispatch(X)
def _json_decode(obj: JSONDataType) -> X:
    return X(obj['value'])


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
        x = X(3)

        encoded = json.dumps(json.serialize(x))
        decoded = json.deserialize(json.loads(encoded))

        self.assertEqual(decoded.value, 3)


def test_json_encode_basic_types() -> None:
    assert json.encode(None) is None
    assert json.encode(3) == 3
    assert json.encode(3.14) == 3.14
    assert json.encode('hello') == 'hello'
    assert json.encode(True) is True


def test_json_encode_compound_types() -> None:
    example_list = [0, 1, 'hello']
    assert json.deserialize(json.serialize(example_list)) == example_list

    example_tuple = (3, 'hi', ('no', 'yes'), True, None)
    assert json.deserialize(json.serialize(example_tuple)) == example_tuple
