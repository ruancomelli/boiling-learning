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
