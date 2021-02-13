from collections.abc import MutableMapping

import funcy

from boiling_learning.utils.utils import (
    SimpleRepr,
    SimpleStr,
    simple_pprint_class
)


@simple_pprint_class
class Parameters(MutableMapping, SimpleRepr, SimpleStr):
    @staticmethod
    def get_from_set(d, key):
        return {k: d[k] for k in key}

    @staticmethod
    def get_from_list(d, key):
        d_ = d
        for k in key:
            d_ = d_[k]
        return d_

    @staticmethod
    def get_from_dict(d, key):
        return {
            v: d[k]
            for k, v in key.items()
        }

    @staticmethod
    def set_from_set(d, key, value):
        for k in key:
            d[k] = value

    @staticmethod
    def set_from_list(d, key, value):
        paths = [[]]
        for k in key:
            if isinstance(k, set):
                paths = [
                    sublist + [k_]
                    for k_ in k
                    for sublist in paths
                ]
            else:
                for p in paths:
                    p.append(k)
        for p in paths:
            d_ = d
            for p_ in p[:-1]:
                d_ = d_.setdefault(p_, {})
            d_[p[-1]] = value

    @staticmethod
    def set_from_dict(d, key, value):
        for k, v in key.items():
            d[k] = value[v]

    def __init__(self, params=None, config=None):
        if params is None:
            params = {}
        self.params = params

        if config is None:
            config = {
                'get': [
                    (funcy.isa(set), Parameters.get_from_set),
                    (funcy.isa(list), Parameters.get_from_list),
                    (funcy.isa(dict), Parameters.get_from_dict)
                ],
                'set': [
                    (funcy.isa(set), Parameters.set_from_set),
                    (funcy.isa(list), Parameters.set_from_list),
                    (funcy.isa(dict), Parameters.set_from_dict)
                ]
            }
        self.config = config

    def register_get_method(self, pred, method):
        self.config.setdefault('get', []).append((pred, method))

    def register_set_method(self, pred, method):
        self.config.setdefault('set', []).append((pred, method))

    def register_del_method(self, pred, method):
        self.config.setdefault('del', []).append((pred, method))

    def __getitem__(self, key):
        for pred, func in self.config.get('get', ()):
            if pred(key):
                return func(self, key)
        return self.params.__getitem__(key)

    def __setitem__(self, key, value):
        for pred, func in self.config.get('set', ()):
            if pred(key):
                func(self, key, value)
                return
        else:
            self.params.__setitem__(key, value)

    def __delitem__(self, key):
        for pred, func in self.config.get('del', ()):
            if pred(key):
                func(self, key)
                return
        else:
            self.params.__delitem__(key)

    def __iter__(self):
        return self.params.__iter__()

    def __len__(self):
        return self.params.__len__()

    # class Fork(dict):
    #     pass

    # def fork(self, forker_classes=(Parameters,), forker_markers=(Parameters.Fork,), propagate=True):
    # 	forked = False
    # 	forks = dict()
    # 	default = dict()

    #     for key, real_value in self.items():
    #         if (
    #             any(isinstance(real_value, forker_marker) for forker_marker in forker_markers)
    #             or (
    #                 propagate
    #                 and any(isinstance(real_value, forker_class) for forker_class in forker_classes)
    #                 and real_value.forked
    #             )
    #         ):
    #             forked = True
    #             for splitter_key in real_value:
    #                 forks.setdefault(splitter_key, default.copy())[key] = real_value[splitter_key]
    #         else:
    #             default[key] = real_value
    #             for v in forks.values():
    #                 v[key] = real_value
    #     if not forked:
    #         forks = self

    #     return forks
