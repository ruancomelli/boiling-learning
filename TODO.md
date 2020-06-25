# TODO List

- [ ] (TEST REQUIRED) Allow the first experimental file to be Experiment HH-MM.txt (without index) in `run_experiment.py`
- [ ] Refactor code. Ideas:
  - use the `collections` library;
  - use the `operator` library;
  - try to remove unnecessary functions;
  - try to remove unnecessary dependencies;
  - [x] remove `__eq__` in `bl.daq.Device`?
- [ ] Write unit tests.
- [ ] Write a config file, so that configurations are not hard-coded in Python, but in a JSON file.
- [ ] Python Learning: design a preprocessing function that takes a `tf.tensor`. This function should take a batch, preprocess it (possibly using many cores) and then fetch the results. The results should then be saved to disk. Useful links: [TensorFlow guide to data performance](https://www.tensorflow.org/guide/data_performance), [TensorFlow tutorial to image classification](https://www.tensorflow.org/tutorials/images/classification), [TensorFlow tutorial to loading images](https://www.tensorflow.org/tutorials/load_data/images), [TensorFlow guide to building input pipelines](https://www.tensorflow.org/guide/data).
- [x] Move library code to a specific module.
- [ ] Improve the project structure:
  - [ ] `bl` has its own `utils`. But I believe that every package should have their own utils.
  - [ ] `bl.utils` could be split into many utilities submodules.
- [ ] Implement parallelization for `TransformationPipeline`s.
- [ ] Use type annotations where applicable.
- [ ] Can wrappers remove code repetition in crop and shift for the `if image is None: image = imread(in_path)` and the `if out_path is not None: imsave(out_path)`?
- [ ] Implement general function dispatching?
- [ ] Document code.
- [ ] Check for inconsistent input values in many functions, some of which are marked with a comment like `# TODO: use check_value_match`.
- [ ] In the `Case` class, many properties can be converted to cached properties. Python 3.8 is be necessary.
- [ ] Allow the user to choose if only the train data must be modified `ImageDatasetTransformer`.
- [ ] Allow dataset augmentation, not only transformation in `ImageDatasetTransformer`.
- [ ] In `boiling_learning.model.restore`: allow a complete dictionary of important keys, not only epoch. For instance, `keys={'epoch_str': int, 'val_acc': float}`.
- [ ] Implement asynchronous `UserPool`.
- [x] In the `Parameters` class, allow getting `dict` keys. Usage: getting from a `dict` creates a branch:

```python
p = Parameters(params={'a': 0})
assert p[{'a': 'b'}] == {'b': 0}
```

- [x] In the `Parameters` class, allow setting `dict` keys. Usage: setting from a `dict` gets specific values from the source:

```python
p = Parameters(params={'a': 0, 'b': 1})
p[{'a': 'C', 'd': 'F'}] = {'C': 1000, 'F': 2000}
assert p == Parameters(params={'a': 1000, 'b': 1, 'd': 2000})
```

- [ ] In the `Parameters` class, decide what to do if there are `set` keys inside `list` keys when getting.
- [ ] In the `Parameters` class, decide what to do if there are `dict` keys inside `list` keys.
- [ ] In the `Parameters` class, decide what to do when multiple `set`s or `dict`s are found in a `list` key.
- [ ] In the `Parameters` class, allow modified `del` paths. For instance, `del p[['a', 'b']]` should delete `'b'` from `p['a']`.
- [x] Fix apparent error in `utils.remove_duplicates`.
- [x] Refactor `Manager`, including separate functions to:
  - [x] create new model path;
  - [x] retrieve existing model path;
  - [x] check if model path already exists;
  - [x] check if model was already saved, or if it exists only in the table;
- [ ] Format code in PEP style.
- [x] DECIDED: it will not. | Decide if `Parameters` will supporting forking.
- [ ] A `dict` can be constructed via a call `dict(...)` or via the special syntax `{k: v ...}`. Support this in `Parameters`? How?
- [ ] Use [TensorFlow estimators](https://www.tensorflow.org/guide/estimator)?
- [x] Allow different batch sizes for different models.
- [ ] Why do `more_itertools.filter_except` and `more_itertools.map_except` need to do `exceptions = tuple(exceptions)`?
- [ ] Finish step detection analysis.
- [ ] Implement a mini-translator: translates from the old-string-format minilanguage to the new format.
- [ ] Implement a function wrapper that transforms the arguments before forwarding. For instance: 

```python
import operator
lower_eq = transform(operator.eq, keyfunc=lambda x: x.lower())
assert 'Hi' != 'hi'
assert lower_eq('Hi', 'hi')
```

- [ ] Why is there a `more_itertools.prepend`, but not a `more_itertools.append`?
- [ ] Am I normalizing images correctly? Make sure I am!
- [x] Allow better querying of models
- [ ] Add another key to a model entry: `state`. For now, I can think of four states: `non-existent`, when the index is not in the lookup table; `empty`, models that were only added to the lookup table but didn't start training; `incomplete`, models that started training; `complete`, models that finished training; `aborted`, models that are not good enough, and are not worth training.
- [ ] Are there better storage methods? For instance, using HDF5 instead of plain .png files? According to [this tutorial](https://realpython.com/storing-images-in-python/), HDF5 is much faster for writing and reading. This would be extremely useful for reading datasets and training. 
- [ ] Write READMEs for each package.
- [ ] Include licenses in each module.
- [ ] Decide if I'll use MLFlow or similars.
- [ ] Remove `process_data.py`, taking care to provide its funciontalities elsewhere.
- [ ] Make `cv2` path-like compliant.
- [ ] Take a look at the relationship between bubble or droplet formation rate and camera acquisition speed.
- [ ] Divide this TO-DO list into sections. For instance: `Refactoring`, `Additional functionality`, `External dependencies` etc.
- [ ] Implement a typing helper `Sentinel` which expects a sentinel value called, for instance, `_sentinel`, or another type. Equivalent to `typing.Optional`, but using any other sentinel instead of `None`. See `typing.Literal` in Python 3.8.
- [ ] There is a similar underlying logic between `bl.model.definitions.utils.ProblemType` and `bl.model.model.SubsetSplit`. This can be converted to a factory function `make_enum(enum_name, enum_values, conversion_table)`, providing a class equivalent to:

```python
  class enum_name:
    enum_values[0] = enum.auto()
    ...
    enum_values[-1] = enum.auto()

    @classmethod
    def get_item(cls, key, default=_sentinel):
        if key in cls:
            return key
        else:
            return cls.from_key(key, default=default)

    @classmethod
    def from_key(cls, key, default=_sentinel):
        for k, v in cls.conversion_table.items():
            if key in v:
                return k
        else:
          if default is _sentinel:
              raise ValueError(f'string {key} was not found in the conversion table. Available values are {list(cls.conversion_table.values())}.')
          else:
              return default

  enum_name.conversion_table = conversion_table
```

- [ ] Create now my own models and tests Kramer's. Some steps are:
  - [ ] Learn where to put Dropout layers. [This paper is awesome](https://arxiv.org/abs/1207.0580).
  - [ ] Always make the number of dense units a multiple of 8. There is a Tensorflow reference for this, find it.
  - [ ] Check if image sizes should be multiples of 8 as well.
  - [ ] Implement droplet/bubble tracking. See what André Provensi texted me.
  - [ ] Can the wet/dry areas ratio be of use to the nets?
  - [ ] Think of cool names for the nets.
- [x] Fix model creation in the notebooks
- [ ] Instead of creating compound objects, define a `metadata` for `Manager`'s entries. So models could look like

```json
{
  "models": {
    "1.model": {
      "data": {
        "creator": "RohanNet",
        "parameters": {
          "n_units": 100,
          "learning_rate": 1e-3
        }
      },
      "metadata": {
        "path": "case 0/models/1.model",
        "timestamp": "2020-07-16 17:50:08",
        "id": "1.model",
        "status": "complete"
      }
    }
  }
}
```

- [ ] Read [this](https://www.wikiwand.com/en/Fraction_of_variance_unexplained). Am I evaluating models correctly?
- [ ] Include `strategy` as part of a model's description?
- [ ] Consume video frames that are already extracted in `bl.preprocessing.video.extract_frames`
- [ ] Refactor `bl.management.Parameters` into `bl.utils.DeepDict` or something else.
- [ ] `bl.utils.DeepDict` should be refactored. Its internals should look like `deep_path` implemented in the `main_notebook.ipynb` notebook. This means that, internally, this class contains a list in the form:

```python
[
  ('a', 'b', 'c', 'd', 'e'),
  ('a', 'b', 'c', 0, 1),
  ('x', 0, ['value'])
]
```

... or maybe not. Think about it!

- [ ] I think that a class `SyncedDict` would be useful. Maybe it could be a base class, I don't know yet. But we could make it something like:

```python
AutoJsonSaver = SyncedDict(save=lambda self: bl.io.save_json(self, 'my_json.json'))
AutoJsonLoader = SyncedDict(load=lambda: bl.io.load_json('my_json.json'))
AutoSynced = SyncedDict(
  save=lambda self: bl.io.save_json(self, 'my_json.json'),
  load=lambda: bl.io.load_json('my_json.json'))
```

a call to `__getitem__` would first call load, and a call to `__setitem__` or `__delitem__` would automatically call the save. Is this good?

- [ ] Create contexts `bl.utils.worker.UserPool.disabled` and `bl.utils.worker.UserPool.enabled` which satisfy:

```python
with user_pool.enabled(): # disabled()
  my_iter = user_pool.get_iterable(range(10))
```

being equivalent to

```python
prev_state = user_pool.is_enabled
user_pool.enable() # disable()
my_iter = user_pool.get_iterable(range(10))
user_pool.is_enabled = prev_state
```

- [ ] Change `bl.utils.worker.UserPool`:
  - [ ] the `distribute_iterable` method should split the iterable and keep it in a cache. When the user calls a `retrieve` method, then the cache is cleared and the iterable is returned.
  - [ ] it should be possible to enforce certain items to belong to certain users. For instance:

  ```python
  >>> user_pool.distribute_iterable([0, 1, 2, 3, 4], {'user2': [5, 6]})
  >>> print(user_pool.cache)
  {
    'user1': [0, 3],
    'user2': [5, 6, 1],
    'user3': [2, 4]
  }
  >>> my_iter = user_pool.retrieve()
  [2, 4]
  ```

  With this structure, it would be possible to do the following: enable GPU acceleration for User 2 only in Google Colab. Assign the models 0-4 to arbitrary users, but ensure that models 5 and 6 (which hipotetically require GPU) are trained by User 2.

- [ ] Implement callbacks for reporting the history and timestamps of a models' training. This would be useful to compare the training of models, in special execution speed (to allow comparison between CPUs versus GPUs or uniform versus mixed precision).
