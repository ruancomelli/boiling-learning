# TODO List

- [x] (TEST REQUIRED) Allow the first experimental file to be Experiment HH-MM.txt (without index) in `run_experiment.py`
- [ ] Refactor code. Ideas:
  - use the `collections` library;
  - use the `operator` library;
  - try to remove unnecessary functions;
  - try to remove unnecessary dependencies;
  - [x] remove `__eq__` in `bl.daq.Device`?
- [ ] Write unit tests.
- [ ] Write a config file, so that configurations are not hard-coded in Python, but in a JSON file.
- [x] Python Learning: design a preprocessing function that takes a `tf.tensor`. This function should take a batch, preprocess it (possibly using many cores) and then fetch the results. The results should then be saved to disk. Useful links: [TensorFlow guide to data performance](https://www.tensorflow.org/guide/data_performance), [TensorFlow tutorial to image classification](https://www.tensorflow.org/tutorials/images/classification), [TensorFlow tutorial to loading images](https://www.tensorflow.org/tutorials/load_data/images), [TensorFlow guide to building input pipelines](https://www.tensorflow.org/guide/data).
- [x] Move library code to a specific module.
- [ ] Improve the project structure:
  - [ ] `bl` has its own `utils`. But I believe that every package should have their own utils.
  - [ ] `bl.utils` could be split into many utilities submodules.
- [x] Implement parallelization for `TransformationPipeline`s.
- [ ] Use type annotations where applicable.
- [x] (DISCARDED) Can wrappers remove code repetition in crop and shift for the `if image is None: image = imread(in_path)` and the `if out_path is not None: imsave(out_path)`?
- [x] (NO - SEE `decorator`) Implement general function dispatching? See [this](https://docs.python.org/3/library/inspect.html#inspect-signature-object).
- [ ] Document code.
- [ ] Check for inconsistent input values in many functions, some of which are marked with a comment like `# TODO: use check_value_match`.
- [ ] In the `ExperimentImages` class, many properties can be converted to cached properties. Python 3.8 is be necessary.
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
- [ ] Autoformat code.
- [x] DECIDED: it will not. | Decide if `Parameters` will supporting forking.
- [ ] A `dict` can be constructed via a call `dict(...)` or via the special syntax `{k: v ...}`. Support this in `Parameters`? How?
- [x] (NO: currently, the support for parallel processing is limited. Stick to Keras models) Use [TensorFlow estimators](https://www.tensorflow.org/guide/estimator)?
- [x] Allow different batch sizes for different models.
- [ ] Why do `more_itertools.filter_except` and `more_itertools.map_except` need to do `exceptions = tuple(exceptions)`?
- [x] Finish step detection analysis.
- [ ] Implement a mini-translator: translates from the old-string-format minilanguage to the new format.
- [x] Implement a function wrapper that transforms the arguments before forwarding. For instance:

```python
import operator
lower_eq = transform(operator.eq, keyfunc=lambda x: x.lower())
assert 'Hi' != 'hi'
assert lower_eq('Hi', 'hi')
```

- [ ] Why is there a `more_itertools.prepend`, but not a `more_itertools.append`?
- [x] Am I normalizing images correctly? Make sure I am!
- [x] Allow better querying of models
- [ ] Add another key to a model entry: `state`. For now, I can think of four states: `non-existent`, when the index is not in the lookup table; `empty`, models that were only added to the lookup table but didn't start training; `incomplete`, models that started training; `complete`, models that finished training; `aborted`, models that are not good enough, and are not worth training.
- [x] \[Using TensorFlow tensors and datasets\] Are there better storage methods? For instance, using HDF5 instead of plain .png files? According to [this tutorial](https://realpython.com/storing-images-in-python/), HDF5 is much faster for writing and reading. This would be extremely useful for reading datasets and training.
- [ ] Write READMEs for each package.
- [ ] Include licenses in each module.
- [x] \[I will not: there's not enough time nor need for this\] Decide if I'll use MLFlow or similar.
- [x] Remove `process_data.py`, taking care to provide its functionalities elsewhere.
- [ ] Make `cv2` path-like compliant.
- [ ] Take a look at the relationship between bubble or droplet formation rate and camera acquisition speed.
- [ ] Divide this TO-DO list into sections. For instance: `Refactoring`, `Additional functionality`, `External dependencies` etc.
- [x] \[No. Take a look at [sentinel package](https://pypi.org/project/sentinel/) or [PEP 0661](https://github.com/taleinat/python-stdlib-sentinels)\] Implement a typing helper `Sentinel` which expects a sentinel value called, for instance, `_sentinel`, or another type. Equivalent to `typing.Optional`, but using any other sentinel instead of `None`. See `typing.Literal` in Python 3.8.
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
- [x] Instead of creating compound objects, define a `metadata` for `Manager`'s entries. So models could look like

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
- [x] Separate `bl.preprocessing.video.extract_frames` into smaller functions
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

- [x] \[Partially solved with `zict`\] I think that a class `SyncedDict` would be useful. Maybe it could be a base class, I don't know yet. But we could make it something like:

```python
AutoJsonSaver = SyncedDict(save=lambda self: bl.io.save_json(self, 'my_json.json'))
AutoJsonLoader = SyncedDict(load=lambda: bl.io.load_json('my_json.json'))
AutoSynced = SyncedDict(
  save=lambda self: bl.io.save_json(self, 'my_json.json'),
  load=lambda: bl.io.load_json('my_json.json'))
```

a call to `__getitem__` would first call load, and a call to `__setitem__` or `__delitem__` would automatically call the save. Is this good? Take a look at [`zict`](https://zict.readthedocs.io/en/latest/).

- [x] Create contexts `bl.utils.worker.UserPool.disabled` and `bl.utils.worker.UserPool.enabled` which satisfy:

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

- [x] Change `bl.utils.worker.UserPool`:
  - [x] (NO) the `distribute_iterable` method should split the iterable and keep it in a cache. When the user calls a `retrieve` method, then the cache is cleared and the iterable is returned.
  - [x] it should be possible to enforce certain items to belong to certain users. For instance:

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

  With this structure, it would be possible to do the following: enable GPU acceleration for User 2 only in Google Colab. Assign the models 0-4 to arbitrary users, but ensure that models 5 and 6 (which hipothetically require GPU) are trained by User 2.

- [ ] Implement callbacks for reporting the history and timestamps of a models' training. This would be useful to compare the training of models, in special execution speed (to allow comparison between CPUs versus GPUs or uniform versus mixed precision).
- [ ] See [Netron](https://github.com/lutzroeder/netron) for NN.
- [ ] Choose a reasonably performing network and train two versions of it: with and without mixed precision. Measure train time and final validation loss. The training should always be performed in the same conditions (i.e. using GPUs and MirroredStrategy), being the application of mixed precision the only difference between the two nets.
- [ ] Organize datasets and publish them on [Kaggle](https://www.kaggle.com/ruancomelli).
- [ ] Use Regions of Interest (ROI)? For instance, see [this](https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44).
- [ ] Use narrower visualization windows?
- [ ] Take a look at [this](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/#about-histogram_freq-what-are-weight-histograms), on how to use TensorBoard, and at [TensorFlow's guide](https://www.tensorflow.org/tensorboard/get_started).
- [x] Simplify `bl.management.Manager`'s interface. I think it would be simpler to have only a `model_id` function, which returns the ID for a model, and then every other function should accept as input this `id` only.
- [x] Add two functions to `bl.management.Manager`: `shared_dir` and `dir`. The first one returns a directory that is shared by all models. The second one returns a directory exclusive to a model.
- [ ] Allow different predicates in `bl.management.Manager.retrieve_models`: besides `entry_pred`, include `metadata_pred`, `description_pred` and `model_pred`. `model_pred` is applied to the model after it is loaded.
- [ ] Include depth? See

> Elboushaki, A., Hannane, R., Afdel, K., Koutti, L., 2020. MultiD-CNN: A multi-dimensional feature learning approach based on deep convolutional networks for gesture recognition in RGB-D image sequences. Expert Systems with Applications.. doi:10.1016/j.eswa.2019.112829

They have two inputs: a RGB image + a depth, which maps each pixel of an image to a relative distance to the photographer. With a 2D experiment, this would be very important to include a depth map to allow the model to see a different between closer bubbles (that should look bigger) and more distant bubbles (which look smaller).

- [ ] Use object detection.
- [ ] Use transfer learning from one case to another.
- [ ] Implement a way to measure the training time.
- [ ] Implement a warm-up: the first epoch of training (after compiling or restoring) should be discarded to avoid including TF warmup in the training time measurement.
- [ ] Optimize for the activation functions
- [ ] \[Maybe\] create classes `Option` and `Options` to help defining model parameters and connect to TensorBoard's HParams.
- [x] Create a function for flattening `Parameters`'s keys. For instance:

```python
>>> d = Parameters()
>>> d[['a', 'b', 'c']] = 10
>>> d[['a', 'x']] = 'oi'
>>> d[['z']] = {'^': '^', '9': 89}
>>> d_flatten = d.flatten_keys(method=lambda prev, key: '.'.join([prev, key]))
>>> print(d_flatten)
{
  'a.b.c': 10,
  'a.x': 'oi',
  'z': {'^': '^', '9': 89} # or 'z.^': '^', 'z.9': 89
}
```

Solution: use [flatten-dict](https://github.com/ianlini/flatten-dict):

```python
d_flatten = flattendict(
    d,
    reducer='dot',
    enumerate_types=(list, tuple, bl.utils.functional.Pack)
)
```

- [x] For many parameters, and above all for setting key names, how about creating a specialized `dataclasses.dataclass`? For instance, instead of:
```python
class CSVDataset:
  def __init__(self, path: Path, features_columns: Optional[List[str]] = None, target_column: str = 'target'):
    if features_columns is None:
      features_columns = ['image_path']

    X = pd.read_csv(path, columns=features_columns + [target_column])
    self.y = X.pop(target_column)
    self.X = X
```

we could write:

```python
@dataclass(frozen=True, kwargs=True)
class CSVDatasetColumns:
  features_columns: List[str] = field(default_factory=lambda: ['image_path'])
  target_column: str = 'target'

class CSVDataset:
  def __init__(self, path: Path, csv_columns: CSVDatasetColumns = CSVDatasetColumns()):
    X = pd.read_csv(path, columns=csv_columns.features_columns + [csv_columns.target_column])
    self.y = X.pop(csv_columns.target_column)
    self.X = X
```

It may become a little bit more verbose, but it also isolates the logic of parameters. Also, it avoids using string constants directly in the function signature, delegating this responsibility to a helper class.

- [x] (DONE: see `bl.preprocessing.transformers`) Perhaps it would be useful to have a class for automatic descriptions. For instance:

```py
>>> @autodescribe('Random contraster')
... def transformer(minval, maxval):
...  return lambda img: tf.image.random_contrast(img, minval, maxval)

>>> print(describe(transformer(0.2, 8)))
    'Random contraster(0.2, 8)'
>>> transformer(0.2, 8)(img)
    ...
```

- [x] Instead of extracting frames, we could iterate over video frames. This would save extraction time, and we would only need to save the final static part of the dataset. **Loose definition:** a *static dataset* is one for which it makes sense saving. For instance, cropping images is a static operation because we know before-hand the interest region. On the other hand, a *dynamic dataset* is one that has to be generated at each call to the train function. For instance, random transformations. A transformation pipeline may have two parts: the first, static one, and the second, dynamic.
- [x] Make `funcy.rpartial` resemble `functools.partial` so that it accepts `**kwargs`. In fact, the documentation of `functools` provides a recipe for the definition of `partial`, and we can just copy that and reverse the order of some arguments, just like I did in `bl.utils.functional.rpartial`
- [ ] Define a more standard way to use sentinels. Also, think of a `SentinelOptional`, which is almost like `Optional`, except that it is defined as `Union[_SentinelType, T]`.
- [ ] Implement [integrated gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients).
- [ ] Perform the following studies:
  - [x] Influence of batch size
  - [x] Learning curve (metric versus dataset size)
  - [ ] Visualization window size
  - [x] Direct versus indirect visualization
  - [ ] How random contrast (and others) affect image variance, and what does this mean in machine learning?
  - [x] Train on one set, evaluate on another
- [ ] Release `Pack` as a standalone package, including functional programming functionality:

```python
def double(arg):
  return 2*arg

def is_greater_than(threshold):
  return lambda arg: arg > threshold

p = Pack('abc', x=3, y=2)
res = (
  p # sends p
  | double # duplicates all values: Pack('aa', 'bb', 'cc', x=6, y=4)
  | (str.upper, is_greater_than(5)) # applies str.upper to args, is_greater_than(5) to kwargs values
)
print(res) # prints Pack('AA', 'BB', 'CC', x=True, y=False)
```

and think of other things.

- [ ] Study RNNs. Perhaps a network could be fed 3 consecutive images (for instance) to give an output.
- [ ] Take a look at [this](https://buildmedia.readthedocs.org/media/pdf/ht/latest/ht.pdf): Python library for heat transfer.
- [ ] Take a look at [`fastai`'s `fastcore`](https://github.com/fastai/fastcore/tree/master/).
- [ ] Take a look at [`BubCNN`](https://github.com/Tim-Haas/BubCNN).
- [ ] Take a look at [this](https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f): use consecutive images for each output.
- [ ] Use [`tf.keras.layers.TimeDistributed`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed) to handle temporal data!
- [ ] Rescale images before feeding them to the network?
- [ ] Use [Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning)?
- [ ] Check [separable convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [ ] Improve `Pack`, making it more like `Parameters`.
- [ ] PACK: Packs Are Compact Kids?
- [ ] [This ideia](https://github.com/kachayev/dataclasses-tensor) looks amazing, maybe use it?
- [ ] Use [LocallyConnected](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LocallyConnected2D) layers?
- [ ] Check this post to get some ideas: https://pub.towardsai.net/state-of-the-art-models-in-every-machine-learning-field-2021-c7cf074da8b2
- [ ] Perhaps objects could be identified by pipelines? Like this:
```py
pipe = manager.Pipeline(Creator('five', lambda: 5))
pipe.append(Transformer('add', lambda x, y: x+y, y=3))
pipe |= Transformer('multiply', lambda x, y: x*y, y=2)

pprint(pipe.json())
# [
#   {
#     "name": "five",
#     "params": [[], {}]
#   },
#   {
#     "name": "add",
#     "params": [[], {"y": 3}]
#   },
#   {
#     "name": "multiply",
#     "params": [[], {"y": 2}]
#   }
# ]

print(pipe())
# 16
# because 16 == (5 + 3)*2
```