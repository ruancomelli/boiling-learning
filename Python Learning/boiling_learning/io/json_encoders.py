import json

from boiling_learning.utils.functional import Pack


class PackEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Pack):
            return [list(obj.args), dict(obj.kwargs)]
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
