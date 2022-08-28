# TODO List

- \[ \] Write unit tests.
- \[ \] Achieve 100% test coverage.
- \[x\] Python Learning: design a preprocessing function that takes a
  `tf.tensor`. This function should take a batch, preprocess it (possibly using
  many cores) and then fetch the results. The results should then be saved to
  disk. Useful links:
  [TensorFlow guide to data performance](https://www.tensorflow.org/guide/data_performance),
  [TensorFlow tutorial to image classification](https://www.tensorflow.org/tutorials/images/classification),
  [TensorFlow tutorial to loading images](https://www.tensorflow.org/tutorials/load_data/images),
  [TensorFlow guide to building input pipelines](https://www.tensorflow.org/guide/data).
- \[x\] `bl.utils` could be split into many utilities submodules.
- \[x\] Use type annotations where applicable.
- \[ \] Document code.
- \[x\] Allow different batch sizes for different models.
- \[ \] Why do `more_itertools.filter_except` and `more_itertools.map_except`
  need to do `exceptions = tuple(exceptions)`?
- \[x\] Finish step detection analysis.
- \[x\] Implement a function wrapper that transforms the arguments before
  forwarding. For instance:

```python
import operator

lower_eq = transform(operator.eq, keyfunc=lambda x: x.lower())
assert "Hi" != "hi"
assert lower_eq("Hi", "hi")
```

- \[x\] Am I normalizing images correctly? Make sure I am!
- \[ \] Write READMEs for each subpackage.
- \[ \] Include licenses in each module.
- \[ \] Make `cv2` path-like compliant.
- \[ \] Take a look at the relationship between bubble or droplet formation rate
  and camera acquisition speed.
- \[x\] \[No. Take a look at
  [sentinel package](https://pypi.org/project/sentinel/) or
  [PEP 0661](https://github.com/taleinat/python-stdlib-sentinels)\] Implement a
  typing helper `Sentinel` which expects a sentinel value called, for instance,
  `_sentinel`, or another type. Equivalent to `typing.Optional`, but using any
  other sentinel instead of `None`. See `typing.Literal` in Python 3.8.
- \[ \] Create my own models and test Kramer's. Some steps are:
  - \[ \] Learn where to put Dropout layers.
    [This paper is awesome](https://arxiv.org/abs/1207.0580).
  - \[ \] Always make the number of dense units a multiple of 8. There is a
    Tensorflow reference for this, find it.
  - \[ \] Check if image sizes should be multiples of 8 as well.
  - \[ \] Implement droplet/bubble tracking. See what André Provensi texted me.
  - \[ \] Can the wet/dry areas ratio be of use to the nets?
  - \[ \] Think of cool names for the nets.
- \[ \] Read
  [this](https://www.wikiwand.com/en/Fraction_of_variance_unexplained). Am I
  evaluating models correctly?
- \[ \] Include `strategy` as part of a model's description?
- \[x\] Implement callbacks for reporting the history and timestamps of a
  models' training. This would be useful to compare the training of models, in
  special execution speed (to allow comparison between CPUs versus GPUs or
  uniform versus mixed precision).
- \[ \] See [Netron](https://github.com/lutzroeder/netron) for NN.
- \[ \] Choose a reasonably performing network and train two versions of it:
  with and without mixed precision. Measure train time and final validation
  loss. The training should always be performed in the same conditions (i.e.
  using GPUs and MirroredStrategy), being the application of mixed precision the
  only difference between the two nets.
- \[ \] Organize datasets and publish them on
  [Kaggle](https://www.kaggle.com/ruancomelli)?
- \[ \] Use narrower visualization windows?
- \[ \] Take a look at
  [this](https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/#about-histogram_freq-what-are-weight-histograms),
  on how to use TensorBoard, and at
  [TensorFlow's guide](https://www.tensorflow.org/tensorboard/get_started).
- \[ \] Include depth? See

> Elboushaki, A., Hannane, R., Afdel, K., Koutti, L., 2020. MultiD-CNN: A
> multi-dimensional feature learning approach based on deep convolutional
> networks for gesture recognition in RGB-D image sequences. Expert Systems with
> Applications.. doi:10.1016/j.eswa.2019.112829

They have two inputs: a RGB image + a depth, which maps each pixel of an image
to a relative distance to the photographer. With a 2D experiment, this would be
very important to include a depth map to allow the model to see a different
between closer bubbles (that should look bigger) and more distant bubbles (which
look smaller).

- \[ \] Use object detection.
- \[ \] Use transfer learning from one case to another.
- \[x\] Implement a way to measure the training time.
- \[ \] Implement a warm-up: the first epoch of training (after compiling or
  restoring) should be discarded to avoid including TF warmup in the training
  time measurement.
- \[ \] Optimize for the activation functions
- \[x\] For many parameters, and above all for setting key names, how about
  creating a specialized `dataclasses.dataclass`? For instance, instead of:

```python
class CSVDataset:
    def __init__(
        self,
        path: Path,
        features_columns: Optional[List[str]] = None,
        target_column: str = "target",
    ) -> None:
        if features_columns is None:
            features_columns = ["image_path"]

        X = pd.read_csv(path, columns=features_columns + [target_column])
        self.y = X.pop(target_column)
        self.X = X
```

we could write:

```python
@dataclass(frozen=True, kwargs=True)
class CSVDatasetColumns:
    features_columns: List[str] = field(default_factory=lambda: ["image_path"])
    target_column: str = "target"


class CSVDataset:
    def __init__(
        self, path: Path, csv_columns: CSVDatasetColumns = CSVDatasetColumns()
    ) -> None:
        X = pd.read_csv(
            path, columns=csv_columns.features_columns + [csv_columns.target_column]
        )
        self.y = X.pop(csv_columns.target_column)
        self.X = X
```

It may become a little bit more verbose, but it also isolates the logic of
parameters. Also, it avoids using string constants directly in the function
signature, delegating this responsibility to a helper class.

- \[ \] Implement
  [integrated gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients).

- \[ \] Perform the following studies:
  - \[x\] Influence of batch size
  - \[x\] Learning curve (metric versus dataset size)
  - \[ \] Visualization window size
  - \[x\] Direct versus indirect visualization
  - \[ \] How random contrast (and others) affect image variance, and what does
    this mean in machine learning?
  - \[x\] Train on one set, evaluate on another
- \[x\] \[No. It is not useful enough.\] Release `Pack` as a standalone package, including functional programming
  functionality:

```python
def double(arg):
    return 2 * arg


def is_greater_than(threshold):
    return lambda arg: arg > threshold


p = Pack("abc", x=3, y=2)
res = (
    p  # sends p
    | double  # duplicates all values: Pack('aa', 'bb', 'cc', x=6, y=4)
    | (
        str.upper,
        is_greater_than(5),
    )  # applies str.upper to args, is_greater_than(5) to kwargs values
)
print(res)  # prints Pack('AA', 'BB', 'CC', x=True, y=False)
```

and think of other things.

- \[ \] Study RNNs. Perhaps a network could be fed 3 consecutive images (for
  instance) to give an output.
- \[ \] Take a look at
  [this](https://buildmedia.readthedocs.org/media/pdf/ht/latest/ht.pdf): Python
  library for heat transfer.
- \[ \] Take a look at
  [`fastai`'s `fastcore`](https://github.com/fastai/fastcore/tree/master/).
- \[ \] Take a look at [`BubCNN`](https://github.com/Tim-Haas/BubCNN).
- \[ \] Take a look at
  [this](https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f):
  use consecutive images for each output.
- \[ \] Use
  [`tf.keras.layers.TimeDistributed`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed)
  to handle temporal data!
- \[ \] Rescale images before feeding them to the network?
- \[ \] Use
  [Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning)?
- \[ \] Check
  [separable convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- \[ \] Improve `Pack`, making it more like `Parameters`.
- \[ \] [This ideia](https://github.com/kachayev/dataclasses-tensor) looks
  amazing, maybe use it?
- \[ \] Use
  [LocallyConnected](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LocallyConnected2D)
  layers?
- \[ \] Check this post to get some ideas:
  https://pub.towardsai.net/state-of-the-art-models-in-every-machine-learning-field-2021-c7cf074da8b2
- \[ \] try model
  [optimization](https://www.tensorflow.org/lite/performance/model_optimization).
- \[ \] try pretrained models from
  [TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1).
- \[ \] try
  [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning).
- \[ \] try to fix the
  [RSquare](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/RSquare)
  metric
- \[ \] implement
  [structural similarity](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity).
  Check the
  papers[(1)](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf),
  [(2)](https://click.endnote.com/viewer?doi=10.1007/s10043-009-0119-z&route=2),
  [(3)](https://www.sciencedirect.com/science/article/pii/S0047259X06002016),
  [(4)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1284395&casa_token=9dJeuWLuFmUAAAAA:J9E9XP0EJerPQXoMVDOqMmpZ_tYsTy4Ig8LgUKjVBD17awcC4aAMEufCS2APZj9BmmUmbWjDk6U&tag=1).
  There is also a tutorial in
  [PythonMachineLearning](https://pythonmachinelearning.pro/structural-similarity-tutorial/).
- \[ \] check
  [Sample Correlation Coefficient](https://www.sciencedirect.com/topics/mathematics/sample-correlation-coefficient).
- \[ \] use [ONNX format](https://onnx.ai/index.html),
  [simplifier](https://github.com/daquexian/onnx-simplifier) and
  [optimizer](https://github.com/onnx/optimizer)?
- \[ \] take a look at the beautiful
  [ConvLSTM2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D)
  for timeseries of images!
- \[ \] take a look at [`ndindex`](https://quansight-labs.github.io/ndindex/index.html)
- \[ \] take a look at [Probabilistic Layers Regression](https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression)
- \[ \] make [LeakyReLU trainable](https://www.tensorflow.org/guide/intro_to_modules#the_build_step)?
- \[ \] [quantize models](https://www.tensorflow.org/model_optimization/guide/quantization/training_example#clone_and_fine-tune_pre-trained_model_with_quantization_aware_training)?
