## v0.17.8 (2021-11-19)

### Fix

- **io/json**: remove wrong `indent` argument in call to `json.load`

## v0.17.7 (2021-11-19)

### Feat

- **datasets/sliceable**: use the extended json (de)serializer for sliceable datasets IO operations

## v0.17.6 (2021-11-19)

### Feat

- implement JSON encode and decode functions for Numpy arrays

## v0.17.5 (2021-11-18)

### Feat

- **scripts/make-dataset**: automatically decide saver and loader functions depending on the `as_tensors` flag

## v0.17.4 (2021-11-18)

## v0.17.3 (2021-11-18)

## v0.17.2 (2021-11-18)

### Fix

- revert creation of `TypedDict`s since they have problems with instance checking

### Feat

- **datasets/sliceable**: define saving and loading functions for sliceable datasets

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

- **datasets/sliceable**: fix `SliceableDataset.split` automatic size scaling to take rounding effects into account

## v0.16.4 (2021-11-16)

### Feat

- **preprocessing/transformers**: add a `__name__` attribute to `Transformer`s

## v0.16.3 (2021-11-16)

### Fix

- **datasets/creators**: avoid forwarding `num_shards` and `snapshot_path` to `experiment_video_dataset_creator` when `as_tensors=False`

## v0.16.2 (2021-11-16)

### Fix

- **main**: fix formatting in `main.py`
- **scripts/make-dataset**: include `num_shards` only when using `as_tensors=True` in `make_dataset.main`

### Refactor

- **management/allocators**: add smart cache tables for faster allocators

## v0.16.1 (2021-11-15)

### Fix

- **merge**: fix merge conflict

## v0.16.0 (2021-11-15)

### Feat

- **scripts/make-dataset**: support making datasets by using arrays instead of tensors
- **datasets/sliceable**: add splitting functionality to sliceable datasets
- **management/descriptors**: add `descriptors` module for automatically describing objects
- **scripts**: add support for processing condensation data using array transformers
- **preprocessing/arrays**: include all data preprocessing functions in the `preprocessing.arrays` module
- **preprocessing/arrays**: convert array preprocessing functions with `@transformer`
- **preprocessing/arrays**: redefine image preprocessing functions in terms of NumPy arrays
- **utils/mathutils**: add `Real` type to denote real number types

### Refactor

- **daq/Channel**: improve type annotations in `daq` module
- improve type annotation and class nomenclature across project
- **utils**: improve `KeyedDefaultDict` type annotations
- **preprocessing/transformers**: improve transformers names, dropping image-exclusive terminology
- **preprocessing/transformers**: improve type annotations in `preprocessing.transformers` module
- **preprocessing/transformers**: remove unused `ImageTransformer.as_image_transformer` method

### Fix

- **datasets/sliceable**: fix `SupervisedSliceableDataset.{features|targets}` type annotations
- **utils/sentinels**: fix import error from `utils.sentinels.EMPTY`

## v0.15.7 (2021-10-18)

### Fix

- **utils/dtypes**: capture `AttributeErrors` in `auto_dtype`

## v0.15.6 (2021-10-18)

### Fix

- **utils/dtypes**: define new `auto_dtype` function, an equivalent to `auto_spec` specialized for dtypes
- **preprocessing/transformers**: replace `auto_spec` with `auto_dtype` in transformer `as_tf_py_function` method for backward compatibility

### Feat

- **datasets/sliceable**: define sliceable datasets based on slicerators and almost compatible with TensorFlow datasets

### Refactor

- **utils/dtypes**: improve type annotations in `utils.dtypes` module

## v0.15.5 (2021-10-18)

### Fix

- **preprocessing/image**: change required dtype for `crop` from float32 to float64

## v0.15.4 (2021-10-17)

### Fix

- **preprocessing/image**: change autocast dtypes to float32 everywhere except `downscale`, for which we keep float64

## v0.15.3 (2021-10-16)

### Feat

- **preprocessing/image**: automatically convert image-like input to float64 `tf.Tensor`s in image manipulation functions

## v0.15.2 (2021-10-15)

### Feat

- **scripts/make-dataset**: accept an optional custom experiment video dataset saver function in `make_dataset` script

### Refactor

- **management/Manager**: improve some parts of the code in `Manager.py` and fix some typing issues

## v0.15.1 (2021-10-14)

### Fix

- **utils/slicerators**: create custom generic `Slicerator`

## v0.15.0 (2021-10-14)

### Feat

- **preprocessing/experiment-video**: accept optional `image_preprocessor` parameter in `ExperimentVideo.as_pairs` to transform images before producing a `Slicerator`
- **typings**: include partial type stubs folder for `slicerator` dependency

### Refactor

- **typing**: add type annotation to `Slicerator`s all over the project
- **datasets/datasets**: remove no-longer-used `Split` class
- **preprocessing/preprocessing**: remove no-longer-used old code for transforming datasets
- **utils/utils**: replace seemingly useless `empty_gen` function with an empty tuple `()`
- **utils/dtypes**: remove unused `print` statements from `auto_spec`
- **utils/dtypes**: remove unused `print` statements from `auto_spec`

### BREAKING CHANGE

- class `Split` is no longer available.
- Old classes for transforming datasets are not available anymore.

### Fix

- **preprocessing/experiment-video**: remove calls to no-longer-existant `ExperimentVideo.save` method

## v0.14.5 (2021-10-12)

### Fix

- **experiment-video**: fix experimental dataframe conversion to records list in `ExperimentVideo.as_pairs`

## v0.14.4 (2021-10-12)

### Refactor

- **datasets**: utilize slicerators instead of datasets in the early setup of experiment video datasets for blazingly fast computations
- **mathutils**: improve type annotation in `mathutils` module

### Fix

- **dataset-creators**: fix bug in `dataset_creator`

### Feat

- **iterutils**: add functions to get indices or masks of evenly spaced items
- **experiment-video**: add `slicerator` functionality to `ExperimentVideo`s for lazy slicing
- **datasets**: make dataset functions `None`- and `DatasetTriplet`-aware
- **datasets**: add `triplet_aware` decorator to make single-dataset functions accept dataset triplets

## v0.14.3 (2021-10-08)

## v0.14.2 (2021-10-08)

## v0.14.1 (2021-10-07)

### Refactor

- **flake8**: move `flake8` configuration to `tox.ini` and delete old `setup.cfg` file

## v0.14.0 (2021-10-07)

### Feat

- **model**: add utility modules for operating with and evaluating models
- **datasets**: define function for bulk-splitting experiment videos

### Fix

- **io**: fix return type from `json_decode(<Path>)`

### Refactor

- **scripts**: move optional imports out of function definition and into the module scope
- **preprocessing**: remove unused `save` parameter in `ExperimentVideo.frames_to_tensor`
- **model-definitions**: comment out old model definitions

### BREAKING CHANGE

- Removed parameter `save` from `ExperimentVideo.frames_to_tensor`.
- BREAKING CHANGE: old models are no longer available through boiling_learning.model._definitions.<model name>

## v0.13.5 (2021-09-29)

### Feat

- **scripts**: include mass rate information in condensation data
- **datasets**: accept optional key in `datasets.targets` function

## v0.13.4 (2021-09-26)

### Feat

- **lazy**: implement right composition between lazy and regular callables

## v0.13.3 (2021-09-26)

### Feat

- **scripts**: accept verbosity flag in `connect_gpus` script and add check for NVIDIA output
- **scripts**: add script for connecting with GPUs

## v0.13.2 (2021-09-26)

### Feat

- **pack**: implement `Pack` partial application using the matrix multiplication operator `@`
- **datasets**: provide decorator for silencing errors when functions are passed `None`s instead of datasets
- **transformers**: add shortcut decorators `creator` and `transformer`

## v0.13.1 (2021-09-14)

### Refactor

- **preprocessing**: move video functionality to its own class, out of `ExperimentVideo`
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

- **io**: fix dispatching bug in JSON (de)serialization and its coupling with table dispatching

### Feat

- **management**: implement disk caching function based on allocators and providers
- **datasets**: add functions for calculating dataset stats and prediction metrics
- **utils**: implement a function for generating context-managed temporary file paths

### Refactor

- **io**: separate JSON encode and decode functions into different methods

## v0.12.0 (2021-08-19)

## v0.11.2 (2021-08-19)

### Feat

- **management**: add utility functions for easier object persistance

## v0.11.1 (2021-08-19)

### Refactor

- simplify code by using Sourcery's suggestions and removing unused functionality
- **scripts**: extract target getting functionality as a function
- import TensorFlow's `AUTOTUNE` from `tensorflow.data` instead of from the experimental module

### Feat

- **datasets**: add `*_flattened` functions, which are shorthand versions of their `*_unbatched` counterparts when the batching key is zero

## v0.11.0 (2021-08-01)

### Fix

- **scripts**: rename main function in `make_boiling_processors` as `main`

### BREAKING CHANGE

- The function `make_boiling_processors` was renamed as `main`.

## v0.10.1 (2021-07-31)

### Feat

- **scripts**: add script for making boiling dataset preprocessors and data augmentors
- **scripts**: implement script for creating preprocessors and data augmentors for condensation datasets
- **scripts**: add script for programmatically making datasets
- **scripts**: add script for programmatically creating, compiling and fitting models

### Refactor

- **scripts**: add a custom runtime error to `make_condensation_processors` in case users try to execute it as a standalone script

## v0.10.0 (2021-07-23)

### Feat

- **scripts**: accept `figsize` parameter to configure figure size directly in consecutive frames analysis
- **scripts**: accept custom frames indexing in consecutive frames analysis

### Refactor

- **scripts**: add legend to downsampling analysis plots
- **scripts**: improve downsampling evaluation script with a better plot and more customization options

### BREAKING CHANGE

- The argument `frames` is now an iterable of tuples `(index, frame)` instead of being simply an iterable of frames. Code using previous versions can be updated by replacing `main(frames, ...)` with `main(enumerate(frames), ...)`.

## v0.9.6 (2021-07-22)

### Fix

- **preprocessing**: allow omitting `frames_tensor_path` when converting `ExperimentVideo`s to datasets without saving

## v0.9.5 (2021-07-20)

### Feat

- **scripts**: define script for setting video data for condensation datasets
- **scripts**: define `set_boiling_cases_data` script for adding video data to boiling cases
- **scripts**: add default error when trying to execute `load_cases` and `load_dataset_tree` as standalone scripts

## v0.9.4 (2021-07-20)

### Feat

- **scripts**: add script for loading a video dataset tree in the form case:subcase:test:video

## v0.9.3 (2021-07-20)

### Feat

- **scripts**: define script for loading cases

### Refactor

- pre-import common subpackages directly on the package's __init__ file

## v0.9.2 (2021-07-14)

### Fix

- **visualization**: avoid `IndexError`s when interacting frames by correctly limiting the frames `IntSlider` widget

## v0.9.1 (2021-07-14)

### Fix

- **visualization**: fix image scaling when forwarding them to `cv2.imshow`

### Refactor

- **preprocessing**: remove old commented code from `ExperimentVideo` source file

## v0.9.0 (2021-07-14)

### Fix

- **preprocessing**: remove competitive behaviour between `ExperimentVideo.open_video` and `ExperimentVideo.data.setter` when defining the end index

### Refactor

- **management**: remove unused `Persistent` functionality

### BREAKING CHANGE

- Classes `Persistent` and `PersistentTransformer` are no longer defined

## v0.8.2 (2021-07-13)

### Fix

- **preprocessing**: use built-in `round` instead of trying (and failing) to import from `math`

## v0.8.1 (2021-07-13)

### Feat

- **preprocessing**: allow setting experiment video `start` and `end` in `VideoData`

## v0.8.0 (2021-07-12)

### Feat

- **preprocessing**: make `ExperimentVideo`s into `Sequence`s of frames

### BREAKING CHANGE

- Previous frame access methods were removed: `.frame()`, `.frames()` and `.sequential_frames()`

### Refactor

- **visualization**: simplify visualization function

## v0.7.1 (2021-07-09)

### Fix

- **visualization**: fix Bokeh visualization when inputs are tensors

## v0.7.0 (2021-07-08)

### Fix

- **visualization**: rename visualizer "boiling_region_cropper" as "region_cropper" to allow different use cases

### BREAKING CHANGE

- Transformers named "boiling_region_cropper" are no longer accepted. Please rename them as "region_cropper".

## v0.6.0 (2021-07-08)

### Fix

- **visualization**: rename annotator "boiling_region_cropper" as "region_cropper" to semantically allow non-boiling transformers

### BREAKING CHANGE

- `DictImageTransformer`s named "boiling_region_cropper" are no longer accepted by visualization functions since they now lack annotators. Please rename them as "region_cropper" to get identical functionality as before.

## v0.5.3 (2021-07-08)

### Fix

- **preprocessing**: fix dataframe type conversion when elapsed time column is absent

## v0.5.2 (2021-07-06)

### Feat

- **models**: make `problem` case-insensitive in `LinearModel`

## v0.5.1 (2021-07-06)

### Fix

- **models**: accept the same arguments in `LinearModel` that we accept in the other models

## v0.5.0 (2021-07-05)

## v0.4.0 (2021-07-05)

### Feat

- **models**: add a LinearModel creator for easier baseline model instantiation

## v0.3.5 (2021-06-30)

### Refactor

- **scripts**: improve output from consecutive frames and downsampling analysis scripts

## v0.3.4 (2021-06-28)

### Feat

- **scripts**: accept a *xscale* parameter in analyses scripts

## v0.3.3 (2021-06-28)

### Feat

- **transformers**: refactor KeyedImageDatasetTransformer for better compatibility with DictImageTransformer

### BREAKING CHANGE

- KeyedImageDatasetTransformer parameter *pack_map* renamed to *packer*

### Fix

- **git**: fix bug in commitizen configuration that still allowed the MAJOR version to increase

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

- **image-preprocessing**: rename *_image_variance* as *variance*

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
