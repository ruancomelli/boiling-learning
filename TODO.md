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
