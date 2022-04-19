## v0.20.6 (2022-04-18)

### Refactor

- **scripts**: reduce need for pre-computing experiment video dataframes

### Fix

- **scripts/set_condensation_datasets_data**: fir `lru_cache` by explicitly adding `maxsize=None`

## v0.20.5 (2022-04-18)

### Refactor

- **scripts/set_condensation_datasets_data**: re-order functions and fix typing
- **scripts/set_condensation_datasets_data**: split monolith into small functions
- simplify scripts-related functionality
- **preprocessing**: simplify pre-processing module
- **daq**: simplify DAQ module

## v0.20.4 (2022-04-15)

### Refactor

- **scripts/set_boiling_cases_data**: refactor module and split functions

### Feat

- **datasets/sliceable**: allow memory caching in `sliceable_dataset_to_tensorflow_dataset`

## v0.20.3 (2022-04-15)

### Feat

- **model/callbacks**: make deletion option on training end in `BackupAndRestore`

### Refactor

- **main**: align functionality with Google Colab script

## v0.20.2 (2022-04-15)

### Feat

- **model/callbacks**: add adaptation for `BackupAndRestore` that correctly removes the backup without errors
- **model/callbacks**: add total training time info to `TimePrinter`

## v0.20.1 (2022-04-15)

## v0.20.0 (2022-04-15)

### Refactor

- **preprocessing/experiment_video**: add logging to `ExperimentVideo.load_df`
- remove unused functions and simplify code
- **io**: remove `io.io` submodule
- remove unused functions and apply minor refactoring
- remove unused functions and variables
- **model/training**: remove unnecessary logging

## v0.19.18 (2022-04-05)

### Refactor

- **model/training**: add logging to `_anonymize_model`

## v0.19.17 (2022-04-05)

### Fix

- **model/training**: fix model anonymization

## v0.19.16 (2022-04-05)

## v0.19.15 (2022-04-05)

### Fix

- **model/training**: anonymize model layers to allow re-usable descriptions
- **model/callbacks**: fix `TimePrinter` logged epoch to use natural numbering

### Refactor

- **model/model**: simplify typing and remove further unnecessary custom functions
- simplify typing and remove further unnecessary custom functions
- **model/callbacks**: simplify `AdditionalValidationSets` by getting rid of cases we do not cover here

## v0.19.14 (2022-04-04)

## v0.19.13 (2022-04-04)

### Feat

- **management/cacher**: add `load` and `save` methods to `Cacher` and `CachedFunction`

### Refactor

- **scripts/analyze_downsampling**: simplify analysis function
- **scripts/analyze_consecutive_frames**: simplify analysis function

## v0.19.12 (2022-03-27)

### Refactor

- replace all logging with `loguru`
- remove type cluttering

## v0.19.11 (2022-03-26)

### Feat

- add logging thanks to `loguru`

## v0.19.10 (2022-03-26)

### Fix

- **sliceable**: fix mapping function to avoid using `slicerator.pipeline`

### Refactor

- **scripts**: improve typing and importing
- improve typing for some functions from `funcy`
- simplify portions of the code
- **preprocessing**: apply minor refactorings

## v0.19.9 (2022-03-21)

## v0.19.8 (2022-03-21)

### Refactor

- apply minor refactorings

## v0.19.7 (2022-03-15)

## v0.19.6 (2022-03-15)

### Feat

- support distribute strategy

### Refactor

- **utils/descriptions**: simplify description overloads to consider only type, and not contents

## v0.19.5 (2022-03-14)

## v0.19.4 (2022-03-14)

### Refactor

- **model**: update model definitions to match the rest of the project
- apply minor refactorings and remove unused functions

## v0.19.3 (2022-03-10)

### Refactor

- remove more unnecessary functionality
- merge array preprocessing module into the image module
- reorganize management module

## v0.19.2 (2022-03-10)

### Refactor

- work around unnecessary dependencies and remove them

## v0.19.1 (2022-03-09)

### Refactor

- remove further code
- remove `Manager`s
- remove more unused functionality
- remove `yogadl` dependency

## v0.19.0 (2022-03-08)

### Refactor

- remove loads of dead code
- remove dataset creators & scripts `make_dataset` and `make_model`
- **main**: clean up old code from main

### BREAKING CHANGE

- removed multiple functions, classes and modules.

## v0.18.24 (2022-03-07)

### Feat

- **model/training**: accept dataset cache dir when fitting model
- **datasets/sliceable**: accept snapshot path in conversion from sliceable datasets to tensorflow datasets

### Refactor

- improve type-annotations and match main script with Google Colab

## v0.18.23 (2022-02-22)

### Feat

- **datasets**: add option to batch, shuffle and prefetch datasets constructed from sliceable datasets

## v0.18.22 (2022-02-21)

### Fix

- fix deprecation warning from TensorFlow by using `tf.keras.callbacks.BackupAndRestore`
- **preprocessing/arrays**: fix albumentations deprecation warning by using `ImageCompression`
- **preprocessing/arrays**: fix albumentations deprecation warning by using `RandomBrightnessContrast`

## v0.18.21 (2022-02-16)

### Fix

- fix end-to-end model training pipeline

### Feat

- **utils/described**: define special `Described` cases

## v0.18.20 (2022-02-14)

### Fix

- fix training pipeline from end-to-end

## v0.18.19 (2022-02-14)

### Feat

- **io**: make `abc.ABCMeta` types JSON serializable

### Refactor

- **io**: add `.json` suffix to JSON saved/loaded files

## v0.18.18 (2022-02-09)

### Fix

- **io**: fix wrong assumption about metadata objects being dicts

## v0.18.17 (2022-02-09)

### Fix

- **io**: fix wrong assumption about metadata objects being dicts

## v0.18.16 (2022-02-08)

### Fix

- **io**: fix wrong import from typing instead of typing-extensions

### Refactor

- **main**: improve structure of `main` script

## v0.18.15 (2022-02-08)

### Feat

- **model/training**: separate cacheable from non-cacheable training functions
- **model**: define save/load functions for Keras models

## v0.18.14 (2022-02-05)

### Feat

- **io**: define save/load operations for sliceable datasets
- **io**: allow save/load basic types and JSON serializable types
- **io/storage**: allow additional metadata in save/load

## v0.18.13 (2022-02-05)

### Refactor

- remove unused version information
- **io**: remove unused functions

## v0.18.12 (2022-02-04)

### Fix

- **datasets/sliceable**: remove unused `filter` method

### Feat

- **datasets/sliceable**: define save/load functions for `SliceableDataset`s
- **io/storage**: define save/load functions for numpy arrays

## v0.18.11 (2022-02-02)

### Feat

- define module for storage of arbitrary objects

### Fix

- fix Makefile to use PDM

### Refactor

- **io**: improve testing and remove unused function

## v0.18.10 (2022-01-30)

### Fix

- **model/training**: apply batching only right after training

## v0.18.9 (2022-01-30)

### Fix

- remove Python 3.8+ usage of `Fraction.as_integer_ratio`

## v0.18.8 (2022-01-28)

### Fix

- fix broken frozendict JSON encoding
- fix training pipeline from end to end

## v0.18.7 (2022-01-25)

### Fix

- **main**: fix dataset generation pipeline
- **io/json**: fix JSON decoding of `Pack`s
- fix I/O and description issues
- **model/training**: import `TypedDict` from `typing_extensions`
- **io/json**: fix dispatching order error

### Refactor

- **io/json**: simplify JSON serialization of lists
- **datasets/sliceable**: make strict zip an option
- **preprocessing/video**: reduce number of video shrinking operations for easier tracking of frames origin
- **preprocessing**: use more `KeyedSet` functionality
- **preprocessing/image_dataset**: make `ImageDataset`s instances of `KeyedSet`s
- **preprocessing/cases**: remove unused parameter from `Case.sync_time_series`
- **models**: revert module naming
- rename module `descriptors` as `descriptions`
- rename subpackage `model` as `models`

### Feat

- **preprocessing/image_datasets**: add description for `ImageDataset`s
- **utils/table_dispatch**: allow dispatching based on predicates
- implement full training pipeline designed for caching
- **utils**: define `Described` objects

## v0.18.6 (2022-01-17)

### Fix

- **preprocessing/experiment_video**: remove incorrect access to removed property

## v0.18.5 (2022-01-14)

### Fix

- **utils/frozendict**: fix support for generic frozendicts

### Refactor

- remove `phantom` dependency (#25)
- **management/cache**: refactor caching out into classes
- simply `preprocessing` model to remove extracted frames
- remove frames extracting functionality
- remove further unused functionality
- remove unused functionality

### Feat

- **main**: add cached function for defining sliceable datasets from image datasets

## v0.18.4 (2022-01-11)

## v0.18.3 (2022-01-11)

## v0.18.2 (2022-01-11)

### Fix

- **management/allocators/json_allocator**: stop recursive calls in `json_allocator` instance check

### Refactor

- refactor project

## v0.18.1 (2022-01-09)

### Refactor

- improve project organization and typing

## v0.18.0 (2022-01-07)

### Fix

- fix main script

### BREAKING CHANGE

- Various functions had to change in a backwards-incompatible way for the fix to be implemented.

## v0.17.27 (2022-01-05)

## v0.17.26 (2022-01-05)

## v0.17.25 (2022-01-04)

### Fix

- **scripts**: fix import error in \`make_model\` script

### Refactor

- remove unused functionality
- improve type-checking across the entire project
- **preprocessing/transformers**: improve exception types in `preprocessing/transformers` module

### BREAKING CHANGE

- many functions and classes have been removed. Client code may have been broken by this.

## v0.17.24 (2021-12-09)

### Fix

- **datasets/sliceable**: add `__name__` property to `DictFeatureTransformer`s

## v0.17.23 (2021-12-06)

### Fix

- **datasets/sliceable**: remove buggy `SliceableDataset.filter` method
- **datasets/sliceable**: fix sliceable dataset bool masking

## v0.17.22 (2021-12-06)

### Fix

- **datasets/sliceable**: fix recursion bug in sliceable datasets

## v0.17.21 (2021-12-06)

### Fix

- **datasets/sliceable**: return `None` instead of empty sliceable datasets in experiment video dataset creator

## v0.17.20 (2021-12-06)

### Feat

- **datasets/sliceable**: support `.prefetch` and `.batch` operations on sliceable datasets

## v0.17.19 (2021-12-06)

### Fix

- **datasets/sliceable**: support `num_parallel_calls` argument in `.map` method in sliceable datasets

## v0.17.18 (2021-12-04)

### Fix

- **datasets**: improve \`triplet_aware\` decorator to handle any kind of datasets

## v0.17.17 (2021-12-02)

### Fix

- **datasets/sliceable**: readd numpy arrays and tensorflow tensors decoding functions

## v0.17.16 (2021-12-02)

### Fix

- **scripts/make-dataset**: fix saver and loader in `make_dataset` when using `as_tensors=False`
- **io**: fix typing for expected exceptions in `add_bool_flag`

## v0.17.15 (2021-11-25)

### Fix

- **preprocessing/experiment-video**: open video before trying to access it in experiment video instances

## v0.17.14 (2021-11-25)

### Fix

- **preprocessing**: update client code to the new `preprocessing.video.Video` API

## v0.17.13 (2021-11-24)

## v0.17.12 (2021-11-23)

### Fix

- try to fix the project versioning again
- fix project versioning

### Feat

- **preprocessing/video**: add automatic video shrinking to avoid failing because of empty frames

## v0.17.11 (2021-11-23)

### Feat

- **io/json**: add `tuple` (de)serialization

## v0.17.10 (2021-11-23)

### Fix

- **io/json**: fix JSON encoding and decoding functions

## v0.17.9 (2021-11-21)

### Fix

- **datasets/sliceable**: fix NumPy arrays (de)serialization

## v0.17.8 (2021-11-19)

### Fix

- **io/json**: remove wrong `indent` argument in call to `json.load`

## v0.17.7 (2021-11-19)

### Feat

- **datasets/sliceable**: use the extended json (de)serializer for sliceable
  datasets IO operations

## v0.17.6 (2021-11-19)

### Feat

- implement JSON encode and decode functions for Numpy arrays

## v0.17.5 (2021-11-18)

### Feat

- **scripts/make-dataset**: automatically decide saver and loader functions
  depending on the `as_tensors` flag

## v0.17.4 (2021-11-18)

## v0.17.3 (2021-11-18)

## v0.17.2 (2021-11-18)

### Fix

- revert creation of `TypedDict`s since they have problems with instance
  checking

### Feat

- **datasets/sliceable**: define saving and loading functions for sliceable
  datasets

### Refactor

- **io**: rename `storage` module as `json`

## v0.17.1 (2021-11-17)

### Feat

- **datasets/sliceable**: add `element_spec` property to sliceable datasets

## v0.17.0 (2021-11-16)

### Refactor

- rename modules and improve imports all across the project
- **preprocessing**: rename `preprocessing.Case` module as `preprocessing.cases`
- **management**: rename `management.Manager` module as `management.managers`
- **management**: remove unused `Mirror` class
- **archive**: remove unused archive

### BREAKING CHANGE

- Many modules were renamed.
- Module now has to be referred to by its new name.

## v0.16.5 (2021-11-16)

### Fix

- **datasets/sliceable**: fix `SliceableDataset.split` automatic size scaling to
  take rounding effects into account

## v0.16.4 (2021-11-16)

### Feat

- **preprocessing/transformers**: add a `__name__` attribute to `Transformer`s

## v0.16.3 (2021-11-16)

### Fix

- **datasets/creators**: avoid forwarding `num_shards` and `snapshot_path` to
  `experiment_video_dataset_creator` when `as_tensors=False`

## v0.16.2 (2021-11-16)

### Fix

- **main**: fix formatting in `main.py`
- **scripts/make-dataset**: include `num_shards` only when using
  `as_tensors=True` in `make_dataset.main`

### Refactor

- **management/allocators**: add smart cache tables for faster allocators

## v0.16.1 (2021-11-15)

### Fix

- **merge**: fix merge conflict

## v0.16.0 (2021-11-15)

### Feat

- **scripts/make-dataset**: support making datasets by using arrays instead of
  tensors
- **datasets/sliceable**: add splitting functionality to sliceable datasets
- **management/descriptors**: add `descriptors` module for automatically
  describing objects
- **scripts**: add support for processing condensation data using array
  transformers
- **preprocessing/arrays**: include all data preprocessing functions in the
  `preprocessing.image` module
- **preprocessing/arrays**: convert array preprocessing functions with
  `@transformer`
- **preprocessing/arrays**: redefine image preprocessing functions in terms of
  NumPy arrays
- **utils/mathutils**: add `Real` type to denote real number types

### Refactor

- **daq/Channel**: improve type annotations in `daq` module
- improve type annotation and class nomenclature across project
- **utils**: improve `KeyedDefaultDict` type annotations
- **preprocessing/transformers**: improve transformers names, dropping
  image-exclusive terminology
- **preprocessing/transformers**: improve type annotations in
  `preprocessing.transformers` module
- **preprocessing/transformers**: remove unused
  `ImageTransformer.as_image_transformer` method

### Fix

- **datasets/sliceable**: fix `SupervisedSliceableDataset.{features|targets}`
  type annotations
- **utils/sentinels**: fix import error from `utils.sentinels.EMPTY`

## v0.15.7 (2021-10-18)

### Fix

- **utils/dtypes**: capture `AttributeErrors` in `auto_dtype`

## v0.15.6 (2021-10-18)

### Fix

- **utils/dtypes**: define new `auto_dtype` function, an equivalent to
  `auto_spec` specialized for dtypes
- **preprocessing/transformers**: replace `auto_spec` with `auto_dtype` in
  transformer `as_tf_py_function` method for backward compatibility

### Feat

- **datasets/sliceable**: define sliceable datasets based on slicerators and
  almost compatible with TensorFlow datasets

### Refactor

- **utils/dtypes**: improve type annotations in `utils.dtypes` module

## v0.15.5 (2021-10-18)

### Fix

- **preprocessing/image**: change required dtype for `crop` from float32 to
  float64

## v0.15.4 (2021-10-17)

### Fix

- **preprocessing/image**: change autocast dtypes to float32 everywhere except
  `downscale`, for which we keep float64

## v0.15.3 (2021-10-16)

### Feat

- **preprocessing/image**: automatically convert image-like input to float64
  `tf.Tensor`s in image manipulation functions

## v0.15.2 (2021-10-15)

### Feat

- **scripts/make-dataset**: accept an optional custom experiment video dataset
  saver function in `make_dataset` script

### Refactor

- **management/Manager**: improve some parts of the code in `Manager.py` and fix
  some typing issues

## v0.15.1 (2021-10-14)

### Fix

- **utils/slicerators**: create custom generic `Slicerator`

## v0.15.0 (2021-10-14)

### Feat

- **preprocessing/experiment-video**: accept optional `image_preprocessor`
  parameter in `ExperimentVideo.as_pairs` to transform images before producing a
  `Slicerator`
- **typings**: include partial type stubs folder for `slicerator` dependency

### Refactor

- **typing**: add type annotation to `Slicerator`s all over the project
- **datasets/datasets**: remove no-longer-used `Split` class
- **preprocessing/preprocessing**: remove no-longer-used old code for
  transforming datasets
- **utils/utils**: replace seemingly useless `empty_gen` function with an empty
  tuple `()`
- **utils/dtypes**: remove unused `print` statements from `auto_spec`
- **utils/dtypes**: remove unused `print` statements from `auto_spec`

### BREAKING CHANGE

- class `Split` is no longer available.
- Old classes for transforming datasets are not available anymore.

### Fix

- **preprocessing/experiment-video**: remove calls to no-longer-existant
  `ExperimentVideo.save` method

## v0.14.5 (2021-10-12)

### Fix

- **experiment-video**: fix experimental dataframe conversion to records list in
  `ExperimentVideo.as_pairs`

## v0.14.4 (2021-10-12)

### Refactor

- **datasets**: utilize slicerators instead of datasets in the early setup of
  experiment video datasets for blazingly fast computations
- **mathutils**: improve type annotation in `mathutils` module

### Fix

- **dataset-creators**: fix bug in `dataset_creator`

### Feat

- **iterutils**: add functions to get indices or masks of evenly spaced items
- **experiment-video**: add `slicerator` functionality to `ExperimentVideo`s for
  lazy slicing
- **datasets**: make dataset functions `None`- and `DatasetTriplet`-aware
- **datasets**: add `triplet_aware` decorator to make single-dataset functions
  accept dataset triplets

## v0.14.3 (2021-10-08)

## v0.14.2 (2021-10-08)

## v0.14.1 (2021-10-07)

### Refactor

- **flake8**: move `flake8` configuration to `tox.ini` and delete old
  `setup.cfg` file

## v0.14.0 (2021-10-07)

### Feat

- **model**: add utility modules for operating with and evaluating models
- **datasets**: define function for bulk-splitting experiment videos

### Fix

- **io**: fix return type from `json_decode(<Path>)`

### Refactor

- **scripts**: move optional imports out of function definition and into the
  module scope
- **preprocessing**: remove unused `save` parameter in
  `ExperimentVideo.frames_to_tensor`
- **model-definitions**: comment out old model definitions

### BREAKING CHANGE

- Removed parameter `save` from `ExperimentVideo.frames_to_tensor`.
- BREAKING CHANGE: old models are no longer available through
  boiling_learning.model.\_definitions.<model name>

## v0.13.5 (2021-09-29)

### Feat

- **scripts**: include mass rate information in condensation data
- **datasets**: accept optional key in `datasets.targets` function

## v0.13.4 (2021-09-26)

### Feat

- **lazy**: implement right composition between lazy and regular callables

## v0.13.3 (2021-09-26)

### Feat

- **scripts**: accept verbosity flag in `connect_gpus` script and add check for
  NVIDIA output
- **scripts**: add script for connecting with GPUs

## v0.13.2 (2021-09-26)

### Feat

- **pack**: implement `Pack` partial application using the matrix multiplication
  operator `@`
- **datasets**: provide decorator for silencing errors when functions are passed
  `None`s instead of datasets
- **transformers**: add shortcut decorators `creator` and `transformer`

## v0.13.1 (2021-09-14)

### Refactor

- **preprocessing**: move video functionality to its own class, out of
  `ExperimentVideo`
- **management**: move lazy functionality from management to utils subpackage

### Feat

- **management**: define `LazyCallable`s that return `Lazy` results
- **management**: add a `Lazy` class for lazily-evaluated objects

## v0.13.0 (2021-09-13)

### Fix

- **commitizen**: fix broken release pipeline
- **scripts**: fix `verbose_load` parameters in scripts

### Feat

- **datasets**: accept `Fraction`al dataset sizes in experiment video creators
- **datasets**: accept `Fraction`al dataset sizes in dataset creators
- **management**: add JSON cacher for easier caching of simple types
- **io**: implement JSON encoding and decoding for `pathlib.Path`s
- **models**: add callback for saving and restoring history at each epoch
- **scripts**: define global script for running the entire code

### Refactor

- **utils**: improve typing in `utils` module
- improve code typing and formatting
- **workers**: remove unused Flask server/client functionality

## v0.12.1 (2021-08-24)

### Fix

- **io**: fix dispatching bug in JSON (de)serialization and its coupling with
  table dispatching

### Feat

- **management**: implement disk caching function based on allocators and
  providers
- **datasets**: add functions for calculating dataset stats and prediction
  metrics
- **utils**: implement a function for generating context-managed temporary file
  paths

### Refactor

- **io**: separate JSON encode and decode functions into different methods

## v0.12.0 (2021-08-19)

## v0.11.2 (2021-08-19)

### Feat

- **management**: add utility functions for easier object persistance

## v0.11.1 (2021-08-19)

### Refactor

- simplify code by using Sourcery's suggestions and removing unused
  functionality
- **scripts**: extract target getting functionality as a function
- import TensorFlow's `AUTOTUNE` from `tensorflow.data` instead of from the
  experimental module

### Feat

- **datasets**: add `*_flattened` functions, which are shorthand versions of
  their `*_unbatched` counterparts when the batching key is zero

## v0.11.0 (2021-08-01)

### Fix

- **scripts**: rename main function in `make_boiling_processors` as `main`

### BREAKING CHANGE

- The function `make_boiling_processors` was renamed as `main`.

## v0.10.1 (2021-07-31)

### Feat

- **scripts**: add script for making boiling dataset preprocessors and data
  augmentors
- **scripts**: implement script for creating preprocessors and data augmentors
  for condensation datasets
- **scripts**: add script for programmatically making datasets
- **scripts**: add script for programmatically creating, compiling and fitting
  models

### Refactor

- **scripts**: add a custom runtime error to `make_condensation_processors` in
  case users try to execute it as a standalone script

## v0.10.0 (2021-07-23)

### Feat

- **scripts**: accept `figsize` parameter to configure figure size directly in
  consecutive frames analysis
- **scripts**: accept custom frames indexing in consecutive frames analysis

### Refactor

- **scripts**: add legend to downsampling analysis plots
- **scripts**: improve downsampling evaluation script with a better plot and
  more customization options

### BREAKING CHANGE

- The argument `frames` is now an iterable of tuples `(index, frame)` instead of
  being simply an iterable of frames. Code using previous versions can be
  updated by replacing `main(frames, ...)` with `main(enumerate(frames), ...)`.

## v0.9.6 (2021-07-22)

### Fix

- **preprocessing**: allow omitting `frames_tensor_path` when converting
  `ExperimentVideo`s to datasets without saving

## v0.9.5 (2021-07-20)

### Feat

- **scripts**: define script for setting video data for condensation datasets
- **scripts**: define `set_boiling_cases_data` script for adding video data to
  boiling cases
- **scripts**: add default error when trying to execute `load_cases` and
  `load_dataset_tree` as standalone scripts

## v0.9.4 (2021-07-20)

### Feat

- **scripts**: add script for loading a video dataset tree in the form
  case:subcase:test:video

## v0.9.3 (2021-07-20)

### Feat

- **scripts**: define script for loading cases

### Refactor

- pre-import common subpackages directly on the package's __init__ file

## v0.9.2 (2021-07-14)

### Fix

- **visualization**: avoid `IndexError`s when interacting frames by correctly
  limiting the frames `IntSlider` widget

## v0.9.1 (2021-07-14)

### Fix

- **visualization**: fix image scaling when forwarding them to `cv2.imshow`

### Refactor

- **preprocessing**: remove old commented code from `ExperimentVideo` source
  file

## v0.9.0 (2021-07-14)

### Fix

- **preprocessing**: remove competitive behaviour between
  `ExperimentVideo.open_video` and `ExperimentVideo.data.setter` when defining
  the end index

### Refactor

- **management**: remove unused `Persistent` functionality

### BREAKING CHANGE

- Classes `Persistent` and `PersistentTransformer` are no longer defined

## v0.8.2 (2021-07-13)

### Fix

- **preprocessing**: use built-in `round` instead of trying (and failing) to
  import from `math`

## v0.8.1 (2021-07-13)

### Feat

- **preprocessing**: allow setting experiment video `start` and `end` in
  `VideoData`

## v0.8.0 (2021-07-12)

### Feat

- **preprocessing**: make `ExperimentVideo`s into `Sequence`s of frames

### BREAKING CHANGE

- Previous frame access methods were removed: `.frame()`, `.frames()` and
  `.sequential_frames()`

### Refactor

- **visualization**: simplify visualization function

## v0.7.1 (2021-07-09)

### Fix

- **visualization**: fix Bokeh visualization when inputs are tensors

## v0.7.0 (2021-07-08)

### Fix

- **visualization**: rename visualizer "boiling_region_cropper" as
  "region_cropper" to allow different use cases

### BREAKING CHANGE

- Transformers named "boiling_region_cropper" are no longer accepted. Please
  rename them as "region_cropper".

## v0.6.0 (2021-07-08)

### Fix

- **visualization**: rename annotator "boiling_region_cropper" as
  "region_cropper" to semantically allow non-boiling transformers

### BREAKING CHANGE

- `DictImageTransformer`s named "boiling_region_cropper" are no longer accepted
  by visualization functions since they now lack annotators. Please rename them
  as "region_cropper" to get identical functionality as before.

## v0.5.3 (2021-07-08)

### Fix

- **preprocessing**: fix dataframe type conversion when elapsed time column is
  absent

## v0.5.2 (2021-07-06)

### Feat

- **models**: make `problem` case-insensitive in `LinearModel`

## v0.5.1 (2021-07-06)

### Fix

- **models**: accept the same arguments in `LinearModel` that we accept in the
  other models

## v0.5.0 (2021-07-05)

## v0.4.0 (2021-07-05)

### Feat

- **models**: add a LinearModel creator for easier baseline model instantiation

## v0.3.5 (2021-06-30)

### Refactor

- **scripts**: improve output from consecutive frames and downsampling analysis
  scripts

## v0.3.4 (2021-06-28)

### Feat

- **scripts**: accept a *xscale* parameter in analyses scripts

## v0.3.3 (2021-06-28)

### Feat

- **transformers**: refactor KeyedImageDatasetTransformer for better
  compatibility with DictImageTransformer

### BREAKING CHANGE

- KeyedImageDatasetTransformer parameter *pack_map* renamed to *packer*

### Fix

- **git**: fix bug in commitizen configuration that still allowed the MAJOR
  version to increase

## v0.3.2 (2021-06-25)

### Refactor

- **datasets**: implement function for applying transformers to a dataset

## v0.3.1 (2021-06-24)

### Fix

- **git**: restage files after pre-commit fixes

## v0.3.0 (2021-06-24)

### Refactor

- fix file endings and sort imports accross project
- update archived ml_final_project script to avoid pre-commit errors
- improve formatting of remaining archived scripts
- remove too old archived scripts

### Feat

- **analyses**: implement consecutive frames analyses

## v0.2.17 (2021-06-24)

### Refactor

- **image-preprocessing**: rename *\_image_variance* as *variance*

### Fix

- downsampling analysis script
- make Shannon cross-entropy numerically stable

## v0.2.16 (2021-06-23)

### Fix

- reshape images to match in *shannon_cross_entropy*

## v0.2.15 (2021-06-23)

### Feat

- implement image statistical comparison methods

## v0.2.14 (2021-06-23)

### Feat

- implement normalized mutual information between images

### Fix

- **docs**: typo in README

## v0.2.13 (2021-06-22)

### Feat

- **simplify-image-datasets**: add more set-like methods to *KeyedSet*
- accept condensation datasets in frame interaction

### Refactor

- improve typing and readability in ImageDataset class definition

## v0.2.12 (2021-06-18)

## v0.2.11 (2021-06-17)

## v0.2.10 (2021-06-17)

## v0.2.9 (2021-06-16)

## v0.2.8 (2021-06-15)

## v0.2.7 (2021-06-14)

## v0.2.6 (2021-06-14)

## v0.2.5 (2021-06-11)

## v0.2.4 (2021-06-10)

## v0.2.3 (2021-06-10)

## v0.2.2 (2021-06-10)

## v0.2.1 (2021-06-10)

## v0.2.0 (2021-06-10)

## v0.1.13 (2021-06-09)

## v0.1.12 (2021-06-09)

## v0.1.11 (2021-06-09)

## v0.1.10 (2021-06-09)

## v0.1.9 (2021-06-09)

## v0.1.8 (2021-06-09)

## v0.1.7 (2021-06-08)

## v0.1.6 (2021-06-08)

## v0.1.5 (2021-06-08)

## v0.1.4 (2021-06-08)

## v0.1.3 (2021-06-08)

## v0.1.2 (2021-06-07)

## v0.1.1 (2021-06-07)
