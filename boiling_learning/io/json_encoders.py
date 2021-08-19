import json
import sys

# class PackEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Pack):
#             return [list(obj.args), dict(obj.kwargs)]
#         # Let the base class default method raise the TypeError
#         return json.JSONEncoder.default(self, obj)


class GenericJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder and Decoder classes to work with general Python classes
    Pass these as the `cls` argument to json.dump and json.load to enable
    the serialization of most Python defined "well behaved"  objects
    directly as JSON.

    Ref.: http://stackoverflow.com/questions/43092113/create-a-class-that-support-json-serialization-for-use-with-celery/43093361#43093361

    Source: https://gist.github.com/rochacbruno/f4d9a0c9c8f712ec31b993034bc5f5a1
    """

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            cls = type(obj)
            return {
                '__custom__': True,
                '__module__': cls.__module__,
                '__name__': cls.__name__,
                'data': obj.__dict__
                if not hasattr(cls, '__json_encode__')
                else obj.__json_encode__,
            }


class GenericJSONDecoder(json.JSONDecoder):
    """Custom JSON encoder and Decoder classes to work with general Python classes
    Pass these as the `cls` argument to json.dump and json.load to enable
    the serialization of most Python defined "well behaved"  objects
    directly as JSON.

    Ref.: http://stackoverflow.com/questions/43092113/create-a-class-that-support-json-serialization-for-use-with-celery/43093361#43093361

    Source: https://gist.github.com/rochacbruno/f4d9a0c9c8f712ec31b993034bc5f5a1
    """

    def decode(self, encoded_str: str):
        result = super().decode(encoded_str)

        if not isinstance(result, dict) or not result.get('__custom__', False):
            return result

        module = result['__module__']
        if module not in sys.modules:
            __import__(module)

        cls = getattr(sys.modules[module], result['__name__'])

        if hasattr(cls, '__json_decode__'):
            return cls.__json_decode__(**result['data'])

        instance = cls.__new__(cls)
        instance.__dict__.update(result['data'])
        return instance
