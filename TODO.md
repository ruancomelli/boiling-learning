# TODO List

- [ ] (TEST REQUIRED) Allow the first experimental file to be Experiment HH-MM.txt (without index) in `run_experiment.py`
- [ ] Use `collections.ChainMap` where this is the intended functionality
- [ ] Write unit tests
- [ ] Write a config file, so that configurations are not hard-coded in Python, but in a JSON file
- [ ] Python Learning: design a preprocessing function that takes a `tf.tensor`. This function should take a batch, preprocess it (possibly using many cores) and then fetch the results. The results should then be saved to disk. Useful links: [TensorFlow guide to data performance](https://www.tensorflow.org/guide/data_performance), [TensorFlow tutorial to image classification](https://www.tensorflow.org/tutorials/images/classification), [TensorFlow tutorial to loading images](https://www.tensorflow.org/tutorials/load_data/images), [TensorFlow guide to building input pipelines](https://www.tensorflow.org/guide/data).
- [x] Move library code to a specific module
- [ ] Improve the project structure
- [ ] Implement parallelization for `TransformationPipeline`s
- [ ] Use type annotations where applicable
- [ ] Can wrappers remove code repetition in crop and shift for the `if image is None: image = imread(in_path)` and the `if out_path is not None: imsave(out_path)`?
- [ ] Document code
- [ ] Check for inconsistent input values in many functions, some of which are marked with a comment like `# TODO: use check_value_match`
- [ ] In the `Case` class, many properties can be converted to cached properties. Python 3.8 may be necessary
- [ ] Allow the user to choose if only the train data must be modified `ImageDatasetTransformer`
- [ ] Allow dataset augmentation, not only transformation in `ImageDatasetTransformer`
- [ ] In boiling_learning.model.restore: allow a complete dictionary of important keys, not only epoch. For instance, `keys={'epoch_str': int, 'val_acc': float}`
- [ ] Implement asynchronous `UserPool`
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
- [ ] In the `Parameters` class, allow modified `del` paths. For instance, `del p[['a', 'b']]` should delete `'b'` from `p['a']`.
- [ ] Fix apparent error in `utils.remove_duplicates`
- [ ] Refactor `ModelManager`, including separate functions to:
  - [ ] create new model path;
  - [ ] retrieve existing model path;
  - [ ] check if model path already exists;
  - [ ] check if model was already saved, or if it exists only in the table;
- [ ] Format code in PEP style
- [ ] Decide if `Parameters` will supporting forking
- [ ] Use [TensorFlow estimators](https://www.tensorflow.org/guide/estimator)?
- [ ] Allow different batch sizes for different models