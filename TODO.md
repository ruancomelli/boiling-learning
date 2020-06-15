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
- [ ] Improve the project structure.
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
- [ ] Devide this TO-DO list into sections. For instance: `Refactoring`, `Additional functionality`, `External dependencies` etc.

