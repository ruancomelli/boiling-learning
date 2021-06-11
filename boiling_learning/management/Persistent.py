# See <https://persistent.readthedocs.io/en/latest/using.html>


class Persistent:
    # structure: file content <----> path <----> value
    # persist (path): value -> file content
    # load (path): file content -> value
    # loaded: condition in which value == file content
    # - True iff:
    #  - loaded or recovered, and:
    #   - value was not modified
    #   - path was not modified
    # free: persist value and then erase value
    # virtual: condition in which value was freed
    # - True iff:
    #  - freed, and:
    #   - value was not modified
    #   - path was not modified
    #  - freed, and:
    #   - used lazy apply
    #   - new path exists
    # persistent: either loaded or virtual

    def __init__(
        self, path, checker=None, writer=None, reader=None, record_paths=False
    ):
        self._record_paths = record_paths
        self.record = []

        self.path = path
        # self._loaded = False
        # self._virtual = False
        self._value = None

        def default_checker(path):
            return path.is_file()

        if checker is None:
            self.checker = default_checker
        else:
            self.checker = checker

        def default_writer(path, value):
            from pickle import dump as pickle_dump

            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open('w') as file:
                pickle_dump(file, value)

        if writer is None:
            self.writer = default_writer
        else:
            self.writer = writer

        def default_reader(path):
            from pickle import load as pickle_load

            with path.open('r') as file:
                return pickle_load(file)

        if reader is None:
            self.reader = default_reader
        else:
            self.reader = reader

    def _append_record(self, new_path):
        if self._record_paths:
            self.record.append(new_path)

    @property
    def value(self):
        self.load()
        return self._value

    @value.setter
    def value(self, other):
        self._loaded = False
        self._virtual = False
        self._value = other

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, other):
        self._loaded = False
        self._virtual = False
        self._path = other
        self._append_record(self._path)

    @property
    def persistent(self):
        return self._loaded or self._virtual

    def free(self):
        self._virtual = True
        self._value = None
        return self

    def persist(self, overwrite=False, writer=None):
        if overwrite or not self.persistent:
            if writer is None:
                writer = self.writer

            writer(self._path, self._value)
            self._loaded = True
            self._virtual = False
        return self

    def load(self, overwrite=False, reader=None):
        if overwrite or not self._loaded:
            if reader is None:
                reader = self.reader

            self._value = reader(self._path)
            self._loaded = True
            self._virtual = False
        return self

    def modify(
        self,
        path_transformer,
        value_transformer,
        share_path=False,
        share_value=False,
        lazy=False,
    ):
        if share_value:
            new_path = path_transformer(self.path, self.value)
        else:
            new_path = path_transformer(self.path)

        if lazy and self.checker(new_path):
            self.path = new_path
            self.free()
        else:
            if share_path:
                self.value = value_transformer(self.path, self.value)
            else:
                self.value = value_transformer(self.value)
            self.path = new_path

        return self


class PersistentTransformer:
    def __init__(self, name, path_transformer, value_transformer, **kwargs):
        self._name = name
        self._path_transformer = path_transformer
        self._value_transformer = value_transformer
        self._kwargs = kwargs

    def __call__(self, persistent):
        return persistent.modify(
            self._path_transformer, self._value_transformer, **self._kwargs
        )
