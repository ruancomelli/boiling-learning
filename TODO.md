# TODO List

- [ ] Allow the first experimental file to be Experiment HH-MM.txt (without index) in `run_experiment.py`
- [ ] Use `collections.ChainMap` where this is the intended functionality
- [ ] Write unit tests
- [ ] Write a config file, so that configurations are not hard-coded in Python, but in a JSON file
- [ ] Python Learning: design a preprocessing function that takes a `tf.tensor`. This function should take a batch, preprocess it (possibly using many cores) and then fetch the results. The results should then be saved to disk
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
