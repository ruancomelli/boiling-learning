import json
from unittest import TestCase

from boiling_learning.io.json import (
    JSONDataType,
    json_decode,
    json_deserialize,
    json_encode,
    json_serialize,
)


@json_encode.dispatch
def _json_encode(obj: tuple) -> list:
    return list(obj)


@json_decode.dispatch(tuple)
def _json_decode(obj: JSONDataType) -> tuple:
    return tuple(obj)


class storage_Test(TestCase):
    def test_default_json_io(self):
        test_list = [
            314159,
            'apple pie tastes good',
            {'likes dinos': True, 'political opinion': None},
        ]

        encoded = json.dumps(json_serialize(test_list))
        decoded = json_deserialize(json.loads(encoded))

        self.assertListEqual(decoded, test_list)

    def test_custom_json_io(self):
        test_tuple = (
            314159,
            'apple pie tastes good',
            {'likes dinos': True, 'political opinion': None},
        )

        encoded = json.dumps(json_serialize(test_tuple))
        decoded = json_deserialize(json.loads(encoded))

        self.assertTupleEqual(decoded, test_tuple)
