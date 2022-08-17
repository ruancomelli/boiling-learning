

## v0.34.12 (2022-08-17)

## v0.34.11 (2022-08-17)

### Refactor

- simplify arguments to `Cropper` in `make_boiling_processors`
- **utils**: simplify type-annotations in `Pack`
- **utils**: use if-guards in `Pack.__getitem__`

## v0.34.10 (2022-08-13)

## v0.34.9 (2022-08-13)

### Feat

- return model evaluation from fitting and autotuning
- accept a crop mode in `make_condensation_preprocessors`

## v0.34.8 (2022-08-09)

### Fix

- **scripts/make_boiling_processors**: remove extra space below wires

## v0.34.7 (2022-08-08)

### Fix

- **datasets/datasets**: fix wrong inheritance in `DatasetTriplet`

## v0.34.6 (2022-08-08)

### Fix

- **scripts**: fix boiling processors to use center crops
- **scripts/plot_steady_state**: wrap script in a function

### Refactor

- **datasets/datasets**: make `DatasetTriplet` a named tuple

## v0.34.5 (2022-08-02)

## v0.34.4 (2022-08-02)

### Fix

- **scripts/make_boiling_processors**: expand bottom for second case

### Feat

- **automl/tuners**: make `goal` optional

## v0.34.3 (2022-07-30)

### Fix

- **scripts/make_boiling_processors**: expand visualization window for second case

### Refactor

- **preprocessing/image**: de-structure shapes instead of index-accessing them
- **scripts**: move steady state plotting script to `scripts` folder
- **daq/channels**: extract variables and merge methods
- **daq/channels**: remove unused utility method `is_type`

## v0.34.2 (2022-07-27)

## v0.34.1 (2022-07-27)

### Refactor

- **utils/pathutils**: remove unused `tempfilepath` and `tempdir`
- complete type annotations
- **utils/dataclasses**: fix typing
- **datasets/metrics**: fix typing
- **utils/table_dispatch**: remove unused `predicate` parameter to `dispatch` method
- **utils/collections**: remove unnecessary `KeyedDefaultDict`
- **utils/collections**: remove unnecessary `merge_dicts`
- **utils/lazy**: remove unused `LazyCallable.__matmul__` operator overload
- **utils/timing**: remove unused `CasesTimer`

## v0.34.0 (2022-07-26)

### Refactor

- **datasets/sliceable**: make `take` and `skip` contiguous and remove `evenly_spaced_indices`
- **utils/dataclasses**: remove unused function `to_parent_dataclass`
- **utils**: remove module `utils.utils`, including unused functions `reorder`, `argmin`, `argmax`, `argsorted` and `onde_factor_at_a_time`

## v0.33.2 (2022-07-25)

### Refactor

- **preprocessing/image_datasets**: improve error message
- **scripts/connect_gpus**: remove unnecessary `else` clauses
- **preprocessing/experiment_video**: remove unnecessary parameter `recalculate` from `make_dataframe`
- **preprocessing/experiment_video**: extract variable in `_sync_dataframes`
- **datasets/sliceable**: reduce usage of explicit type-annotations in `SliceableDataset.split`
- **model/definitions**: rename intermediary layers from `x` to `outputs`

### Feat

- **management/allocators**: support `suffix` parameter to set filename suffix when saving objects with `JSONTableAllocator` and `default_table_allocator`

## v0.33.1 (2022-07-13)

## v0.33.0 (2022-07-13)

### Refactor

- **utils**: create `utils.pathutils` and move path-related functionality into it
- **utils**: create `utils.pathutils` and move path-related functionality into it
- **utils**: remove unnecessary `SimpleStr` and `SimpleRepr` mixin classes

## v0.32.0 (2022-07-12)

### Refactor

- **utils**: move `merge_dicts` to `utils.collections`
- **utils**: move `KeyedDefaultDict` to `utils.collections`

### Feat

- **datasets/cache**: support constructing a numpy cache adjusted to a dataset

## v0.31.0 (2022-07-12)

### Refactor

- **utils**: move `unsort` to `utils.iterutils`
- **preprocessing/image**: remove unused image normalization preprocessing function
- **utils/utils**: remove unused utility function `indexify`
- **model/definitions**: remove unused custom models

### Feat

- **datasets/sliceable**: add support for sliceable dataset caching
- **automl**: define auto-tunable block for image normalization
- **preprocessing/numpy**: make `frames_to_numpy` lazier by not writing to disk and flushing instead of accumulating all data in memory

## v0.30.20 (2022-07-06)

### Fix

- **model/model**: correctly pass `custom_objects` to `tf.keras.Model.from_config`

## v0.30.19 (2022-07-06)

### Fix

- **automl/tuners**: fix inheritance chain

## v0.30.18 (2022-07-06)

### Refactor

- **automl/tuners**: log max model size for easier tracking
- **automl/tuners**: log max model size for easier tracking

## v0.30.17 (2022-07-05)

### Feat

- **automl/hypermodels**: define `ConvImageRegressor` for auto-tuning image regression models with convnets
- **automl/tuners**: define early-stopping Hyperband tuner
- **model/model**: add utility function for renaming models

### Refactor

- **automl/tuners**: remove `goal` from oracle

## v0.30.16 (2022-07-03)

### Fix

- **automl/tuners**: ensure that the checkpoint file parents exist

### Refactor

- **scripts/set_boiling_cases_data**: move wire sample data to `boiling_learning.data.samples`

## v0.30.15 (2022-07-03)

### Fix

- **automl/tuners**: ensure that model weights are read to/from HDF5 files

### Refactor

- **preprocessing/image_datasets**: re-use utility function
- **preprocessing/image_datasets**: simplify `set_video_data`
- **preprocessing/image_datasets**: automatically apply "purge" option

## v0.30.14 (2022-07-03)

### Fix

- **automl/tuners**: do not use beta Keras Tuner functions

### Refactor

- **model/definitions**: remove unused configuration enums
- **datasets/bridging**: move `utils.dtypes` into `datasets/bridging`

## v0.30.13 (2022-07-03)

### Fix

- **automl/tuners**: save models at the end of the training

### Refactor

- **model/definitions**: simplify model definitions to use `Literal`s instead of enums

## v0.30.12 (2022-07-03)

### Fix

- **automl/hypermodels**: use `LayersBlock.get_config` when allocating path for tuner in `FixedArchitectureImageRegressor`

## v0.30.11 (2022-07-02)

### Feat

- **automl/hypermodels**: accept allocators for saving hypermodels

## v0.30.10 (2022-07-02)

### Fix

- **automl/tuners**: skip saving models on each epoch

## v0.30.9 (2022-07-01)

### Refactor

- **preprocessing/image_datasets**: remove unused functionality from `ImageDataset`

### Fix

- **automl/hypermodels**: do not try to get "config" key in `HyperModel.__json_encode__`

## v0.30.8 (2022-07-01)

### Fix

- **model/model**: revert to using HDF5 files to save and load models

## v0.30.7 (2022-07-01)

### Fix

- **io/json**: ensure that nested dictionaries are always sorted

## v0.30.6 (2022-06-30)

### Fix

- **io/json**: ensure that dictionaries are always sorted

## v0.30.5 (2022-06-29)

### Feat

- **preprocessing/video**: support mapping a function to multiple frames at once

## v0.30.4 (2022-06-29)

### Fix

- **preprocessing/image**: fix tiny typing issue
- **model/model**: allow model weights to be saved in TF format

### Feat

- **scripts**: add support for arrays of frames to the processor makers
- **preprocessing/image**: support frames and arrays of images
- **preprocessing/numpy**: add caching of frames datasets using numpy files
- **data/boiling_curve**: add full data for Incropera boiling curve with imposed heat flux
- **data/boiling_curve**: add full data for boiling curve from Incropera
- add `data` subpackage containing experimental data
- **utils/typeutils**: add utility types `Pair` and `Triplet`

### Refactor

- **automl**: simplify hypermodel and tuner classes

## v0.30.3 (2022-06-26)

### Fix

- **automl/tuning**: correctly retrieve `callbacks`

## v0.30.2 (2022-06-26)

### Fix

- **automl/tuning**: default batch size to `32`

## v0.30.1 (2022-06-26)

### Fix

- **automl/tuning**: remove extra field from `TuneModelParams`

### Refactor

- **automl/hypermodels**: make `ImageRegressor` and `FixedArchitectureImageRegressor` subclasses of `Hypermodel`
- **utils/functional**: remove unused function
- **utils/slicerators**: remove no-longer used `slicerator` dependency

### Feat

- **automl/hypermodels**: define `FixedArchitectureImageRegressor` automodel
- **automl/hypermodels**: define `ImageRegressor` automodel

## v0.30.0 (2022-06-26)

### Fix

- **automl/tuners**: suppress errors in `_try_build`
- **automl/blocks**: correctly rename layers in `LayersBlock`
- **automl/tuners**: correctly initialize parents in `_FixedMaxModelSizeGreedy`

### Refactor

- **scripts/set_condensation_datasets_data**: remove unused parameter `datasheet_name`
- **utils/iterutils**: replace enumeration with literal of strings
- **model/automl**: move auto ML functionality to its own subpackage
- **model/model**: rework `anonymize_model_json` to rename deeply nested layers

### Feat

- **automl**: add block for constant architecture
- **model/model**: support `get_config` and `from_config` in `ModelArchitecture`
- **automl**: add support for autotuning models
- **model/automl**: add auto ML tuner

## v0.29.6 (2022-06-19)

### Fix

- **scripts**: move grayscaling back to the first operation to avoid having color distortion near the edges

## v0.29.5 (2022-06-19)

## v0.29.4 (2022-06-19)

### Refactor

- **datasets/sliceable**: make `__repr__` a mandatory method
- **datasets/sliceable**: make `ConcatenateSliceableDataset`s variadic

## v0.29.3 (2022-06-18)

### Feat

- **model/training**: add `load_with_strategy` function for loading models using a specific distributed strategy

## v0.29.2 (2022-06-18)

### Fix

- **scripts/set_condensation_datasets_data**: skip cases for which no data is available

## v0.29.1 (2022-06-18)

### Fix

- **preprocessing/image**: re-use skimage downscale function
- **model/model**: pass custom layers to `tf.keras.models.model_from_json` as `custom_objects`

## v0.29.0 (2022-06-18)

### Feat

- **scripts/set_condensation_datasets_data**: add experiment video case:subcase to its categories
- **preprocessing/transformers**: add `__str__` method to transformers

### Refactor

- **scripts**: remove unnecessary FPS cache

### Fix

- **model/definitions**: normalize images by default
- **scripts**: correctly get FPS from experiment videos
- **scripts**: add step for converting images to `float32`
- **management/allocators**: normalize packs to facilitate description matches

## v0.28.2 (2022-06-17)

### Refactor

- **scripts/run_experiment**: comment out currently unused script

### Feat

- **management/allocators**: add logging to `JSONTableAllocator`

## v0.28.1 (2022-06-17)

### Feat

- **model/layers**: add image normalization preprocessing layer

## v0.28.0 (2022-06-17)

### Refactor

- **datasets/datasets**: remove unused functionality
- **preprocessing/transformers**: remove transformer names

## v0.27.1 (2022-06-17)

### Fix

- **preprocessing/visualize**: remove usage of `DictTransformer`
- **preprocessing/image**: use TF to transform all images
- **preprocessing/image**: correctly convert all transformed data to numpy and squeeze arrays

## v0.27.0 (2022-06-17)

### Refactor

- **preprocessing/transformers**: remove `DictTransformer`
- **scripts**: remove image augmentation steps
- **datasets/sliceable**: replace `SupervisedSliceableDataset` functionality with static functions
- **preprocessing/image**: add transformers for downscaling and grayscaling images

### Feat

- **preprocessing/image**: add `Cropper` transformer
- **datasets/sliceable**: add `repeat` method and `constantly` constructor

## v0.26.7 (2022-06-16)

### Feat

- **datasets/bridging**: support pre-filtering and post-filtering datasets

## v0.26.6 (2022-06-16)

### Feat

- **datasets/bridging**: support getting targets

## v0.26.5 (2022-06-16)

### Fix

- **scripts**: do not convert categories to integers

## v0.26.4 (2022-06-16)

### Feat

- **datasets/bridging**: support caching

## v0.26.3 (2022-06-16)

### Fix

- **datasets/sliceable**: avoid index errors when slicing sliceable datasets

## v0.26.2 (2022-06-16)

### Refactor

- **preprocessing**: improve type-annotations in `image` and `transformers` modules

### Perf

- **scripts**: re-order processors to run the faster and most memory-saving operation first

## v0.26.1 (2022-06-16)

### Fix

- **preprocessing/video**: define `VideoFrameU8` and `VideoFrameF32` even when not type-checking

## v0.26.0 (2022-06-16)

### Fix

- **scripts/make_boiling_processors**: remove conversion to float32
- **scripts**: move random cropper to the "preprocessing" group since it is mandatory to enforce equal image size

### Refactor

- **preprocessing/hdf5**: remove compression option
- **preprocessing/video**: improve `VideoFrame` type annotations
- **preprocessing/video**: replace PIMS with Decord
- remove more unused functionality
- **preprocessing/transformers**: remove unnecessary class `FeatureTransformer` and adapt module
- **preprocessing/transformers**: remove unused class `KeyedFeatureTransformer`

## v0.25.5 (2022-06-15)

### Feat

- **preprocessing/experiment_video**: use `DecordVideo` instead of `PimsVideo`

### Fix

- **preprocessing/experiment_video**: get frame from video, not from experiment-video

## v0.25.4 (2022-06-15)

### Fix

- **scripts**: fix last valid frame getting to support experiment videos

## v0.25.3 (2022-06-15)

### Refactor

- **preprocessing**: remove unnecessary module `preprocessing` by moving its utility functionality to where needed

### Fix

- **preprocessing/experiment_video**: add descriptor and JSON serializer for `ExperimentVideo`

## v0.25.2 (2022-06-15)

### Fix

- **preprocessing/video**: add one environment variable for trying to limit decord memory usage

## v0.25.1 (2022-06-15)

### Fix

- **preprocessing/video**: fix frame scaling to the 0-1 range

## v0.25.0 (2022-06-15)

### Feat

- **preprocessing/video**: implement decord-based video class

### Refactor

- **preprocessing/video**: make `Video` class more compact
- **preprocessing/experiment_video**: prefer composition over inheritance
- **datasets/bridging**: make parameters stable

## v0.24.0 (2022-06-12)

### Refactor

- **preprocessing/hdf5**: use further sliceable dataset methods in `frames_to_hdf5`
- **preprocessing/hdf5**: remove unnecessarily added `indices` parameter
- **preprocessing/hdf5**: rename `batch_size` as `buffer_size` in `frames_to_hdf5`
- **preprocessing/hdf5**: generalize `video_to_hdf5` as `frames_to_hdf5` accepting any dataset of frames
- **preprocessing/hdf5**: make compressing data optional
- **preprocessing/hdf5**: remove SWMR configuration since we will never write to `HDF5SliceableDataset`s
- **preprocessing/hdf5**: make use of the new sliceable interface for videos
- **preprocessing/video**: make `fps` a method in `Video`
- **preprocessing/video**: convert `Video` to a `SliceableDataset`
- **preprocessing/video**: remove unused functionality

### Feat

- **datasets/sliceable**: add `enumerate` method to sliceable datasets

## v0.23.7 (2022-06-11)

### Feat

- **datasets/bridging**: support reading dataset shards in parallel
- **model/model**: add utility methods for counting the number of parameters and the size of a model

### Refactor

- **model/definitions**: set layers policies as the default policy by default
- **model/callbacks**: improve displaying of metrics in `TimePrinter` and `AdditionalValidationSets`

### Fix

- **scripts/set_condensation_datasets_data**: correctly return `None` when the timestamp regex match fails

## v0.23.6 (2022-06-08)

### Fix

- **io/storage**: add `_metadata` to `_deserialize_timedelta`

## v0.23.5 (2022-06-08)

### Fix

- **model/model**: revert making `ModelArchitecture` a dataclass

## v0.23.4 (2022-06-08)

### Fix

- **model/model**: fix model from JSON call

### Refactor

- **model**: wrap TF models in `ModelArchitecture`
- **model**: move `ModelArchitecture` to `boiling_learning.model.model`

### Feat

- **datasets/bridging**: accept experimental parameters for reading datasets in parallel

## v0.23.3 (2022-06-07)

### Fix

- **datasets/bridging**: remove extraneous keyword parameter

## v0.23.2 (2022-06-07)

### Fix

- **datasets/bridging**: correctly capture non-existing save files

## v0.23.1 (2022-06-07)

### Feat

- **datasets/bridging**: make `prefetch` an integer

## v0.23.0 (2022-06-07)

### Refactor

- **datasets/bridging**: remove trivially redundant parameter `shuffle`
- **utils/dtypes**: remove unused functions
- **model/callbacks**: simplify `MemoryCleanUp` to do only garbage collection

### Fix

- **datasets/bridging**: allow safe bridging to TF datasets when using save paths

### Feat

- **io/storage**: support (de)serialization of timedelta objects

## v0.22.3 (2022-06-06)

### Refactor

- **preprocessing/experiment_video**: refactor `targets` method to return a dataframe
- **preprocessing/hdf5**: make HDF5 dataset generic

## v0.22.2 (2022-06-06)

### Feat

- **datasets/bridging**: add support for post-caching filters and mappers

## v0.22.1 (2022-06-06)

### Refactor

- **datasets/sliceable**: rename ancestor datasets as `_ancestor` in `MapSliceableDataset` and `BatchSliceableDataset`
- **datasets/sliceable**: reduce method indirection

### Fix

- **datasets/sliceable**: turn indices into tuple before sending to component datasets in `ZippedSliceableDataset`

## v0.22.0 (2022-06-06)

### Refactor

- **datasets/sliceable**: remove unused IO functions with sliceable datasets

### Fix

- **datasets/sliceable**: remove useless and inefficient `getitem_from_indices` specialization in `ConcatenateSliceableDataset`

## v0.21.3 (2022-06-06)

### Fix

- **model/training**: fix `FitModelReturn` (de)serialization

## v0.21.2 (2022-06-05)

### Refactor

- **datasets/sliceable**: refactor fix for `ConcatenateSliceableDataset`

### Fix

- **datasets/sliceable**: fix `get_from_indices` and `fetch` in `ConcatenateSliceableDataset`
- **model/callbacks**: use natural indexing when registering epochs

## v0.21.1 (2022-06-05)

### Feat

- **model/training**: make the return from `get_fit_model` serializable

### Fix

- **datasets/bridging**: fix type annotation for filters

## v0.21.0 (2022-06-05)

### Refactor

- **datasets/bridging**: re-use `SliceableDataset.element_spec`
- **model/training**: remove cache and snapshot parameters
- **model/callbacks**: avoid printing when evaluating models in `AdditionalValidationSets`

### Fix

- **datasets/bridging**: fix dataset expansion to batch size when there is filtering

## v0.20.41 (2022-06-05)

### Perf

- **datasets/bridging**: cache dataset before batching for easier re-use

### Feat

- support JSON serialization and description of `datetime.timedelta` objects
- **model/definitions**: allow disabling droupout by passing `None`

### Fix

- **model/callbacks**: support numpy floats in `SaveHistory`

### Refactor

- **model/definitions**: replace `Activation("softmax")` with `Softmax()`

## v0.20.40 (2022-06-05)

### Feat

- **model/training**: register training time, end epoch and history

## v0.20.39 (2022-06-04)

## v0.20.38 (2022-06-04)

### Refactor

- **model/training**: remove unnecessary arguments to `model.fit` and use new functionality from the bridging module
- **model/callbacks**: make "mode" mandatory and keyword-only in `SaveHistory`
- **model/callbacks**: remove extraneous arguments
- **model/callbacks**: improve time printer

### Feat

- **datasets/bridging**: add support for shuffling and auto-fixing batch size issues to `sliceable_dataset_to_tensorflow_dataset`
- **datasets/bridging**: add support for filtering in `sliceable_dataset_to_tensorflow_dataset`
- **management/cacher**: add decorator for cacher-aware functions
- **management/allocators**: add support for `allocate` method

### Fix

- **preprocessing/image_datasets**: fix image datasets description to be order-insensitive

## v0.20.37 (2022-06-04)

### Fix

- **preprocessing/hdf5**: ensure correct HDF5 opening and closing on every access to data
- **model/definitions**: fix memory leaks by avoiding stacking convnets without pooling

### Feat

- **model/callbacks**: add memory clean-up callback

### Refactor

- **model/callbacks**: improve typing

## v0.20.36 (2022-06-02)

### Fix

- **model/callbacks**: add error handling to `AdditionalValidationSets`

## v0.20.35 (2022-06-01)

### Fix

- **preprocessing/hdf5**: replace OpenCV reader with PIMS for correct frame generation

## v0.20.34 (2022-06-01)

### Fix

- **utils**: handle empty iterables in `unsort`

## v0.20.33 (2022-06-01)

### Feat

- **preprocessing/hdf5**: support different HDF5 file opening modes

## v0.20.32 (2022-06-01)

### Refactor

- **preprocessing/hdf5**: convert frames to the 0-1 scale before saving to HDF5, not after

## v0.20.31 (2022-05-31)

### Fix

- **preprocessing/hdf5**: sort indices and unsort frames to avoid HDF5 access errors

## v0.20.30 (2022-05-31)

### Fix

- **datasets/sliceable**: unswap strictness checking methods

## v0.20.29 (2022-05-31)

### Feat

- **datasets/sliceable**: accept different strictness levels in `ZippedSliceableDataset`

## v0.20.28 (2022-05-30)

### Fix

- **preprocessing/hdf5**: add `__len__` to `HDF5VideoSliceableDataset`

## v0.20.27 (2022-05-30)

### Feat

- **model/training**: add prefetch to sliceable datasets

## v0.20.26 (2022-05-30)

### Refactor

- **preprocessing/hdf5**: improve sliceable dataset and add logging

### Fix

- **preprocessing/hdf5**: divide image pixels by 255

## v0.20.25 (2022-05-30)

### Feat

- **preprocessing/hdf5**: provide a sliceable dataset based on HDF5 files
- **datasets/sliceable**: add `PrefetchDataset`
- **model/layers**: add backport for layer `RandomBrightness` from TF 2.9

### Fix

- **datasets/sliceable**: fix `SupervisedSliceableDataset` to account for the newest changes

### Refactor

- **datasets/sliceable**: refactor more sliceable dataset functionality into separate dedicated classes
- **datasets/sliceable**: refactor sliceable dataset functionality into separate dedicated classes
- **datasets**: move `sliceable_dataset_to_tensorflow_dataset` to a new `bridging` module

## v0.20.24 (2022-05-28)

### Perf

- **main**: replace individual JSON caching for images with HDF5 caching
- **preprocessing/hdf5**: refactor loop-independent expression out of loop

### Refactor

- **main**: align `main` with code on Google Colab

### Feat

- **proprocessing/hdf5**: add support for transformers of batches to conversion to HDF5
- **preprocessing/video**: add utility function for getting frame from HDF5 file
- **preprocessing/video**: add video conversion method to HDF5

## v0.20.23 (2022-05-25)

### Feat

- **model/training**: correctly accept snapshot and cache paths

## v0.20.22 (2022-05-24)

### Fix

- **utils/descriptions**: use a shallow-copier version of `asdict` to describe and serialize objects to JSON

## v0.20.21 (2022-05-24)

### Fix

- **management/persister**: fix wrong `issubclass` for input exceptions

## v0.20.20 (2022-05-24)

### Refactor

- add logging for saving cached objects
- **preprocessing/video**: add debug logging messages to `valid_end_frame`

## v0.20.19 (2022-05-22)

### Refactor

- **utils/lazy**: simplify implementation of laziness
- **utils/utils**: remove unused function `missing_ints`

## v0.20.18 (2022-05-21)

### Refactor

- **model**: remove unused custom metric `RSquare` and dependency on `typeguard`
- apply minor refactorings
- replace `dataclassy` with the built-in `dataclasses`

## v0.20.17 (2022-05-16)

### Refactor

- **scripts/set_boiling_cases_data**: support caching EV valid frames
- **preprocessing/video**: remove automatic video shrinking
- **datasets/sliceable**: remove unnecessary `shuffle` parameter from `sliceable_dataset_to_tensorflow_dataset`
- **model/definitions**: improve typing in module `model.definitions`
- improve typing accross some bits of the project

## v0.20.16 (2022-05-10)

### Refactor

- **main**: cache EV shrinking length in `main`

## v0.20.15 (2022-05-10)

### Refactor

- **scripts/set_condensation_datasets_data**: cache EV shrinking length

### Feat

- **preprocessing/video**: allow locally disabling automatic video shrinking

## v0.20.14 (2022-05-10)

### Refactor

- **preprocessing/video**: add logging to video shrinking
- **io/json**: move `JSONDataType` to this module, out of `utils.utils`
- **preprocessing**: simplify preprocessing module by removing unnecessary functionality
- **preprocessing**: remove no-longer needed methods for operating with dataframes
- **preprocessing/cases**: remove unused method for syncing dataframes

## v0.20.13 (2022-05-04)

### Refactor

- **preprocessing/experiment_video**: add logging to `ExperimentVideo._shrink_video_to_data`

### Fix

- **main**: set specific random state in main

## v0.20.12 (2022-04-21)

### Fix

- **preprocessing/image**: convert `tf.float64` frames to `tf.float32` in `random_jpeg_quality` because of errors in `albumentations`

### Refactor

- **preprocessing**: apply small refactorings

## v0.20.11 (2022-04-21)

### Refactor

- **preprocessing/experiment_video**: refactor module a bit
- **preprocessing/experiment_video**: refactor `ExperimentVideo` class and remove `as_pairs` method
- **preprocessing/experiment_video**: remove `set_video_data` method

### Fix

- **scripts/set_condensation_datasets_data**: avoid using experiment videos that didn't have their data set

## v0.20.10 (2022-04-21)

### Fix

- **scripts/set_condensation_datasets_data**: fix wrong iteration over `ImageDataset`s

### Refactor

- **preprocessing/video**: remove printing cause exception when opening invalid videos

## v0.20.9 (2022-04-21)

## v0.20.8 (2022-04-21)

### Refactor

- **preprocessing/video**: improve error message when opening invalid videos

## v0.20.7 (2022-04-19)

### Feat

- **scripts/utils**: commit new utility method `setting_data`

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
