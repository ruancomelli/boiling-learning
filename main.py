from operator import itemgetter

import tensorflow as tf
from loguru import logger
from rich.console import Console

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.app.datasets.raw.boiling1d import boiling_data_path
from boiling_learning.app.datasets.raw.condensation import (
    condensation_data_path,
    condensation_datasets,
)
from boiling_learning.app.paths import analyses_path
from boiling_learning.app.training.boiling1d import (
    get_baseline_boiling_architecture,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.datasets.sliceable import map_targets, targets
from boiling_learning.image_datasets import Targets
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.training import CompileModelParams, compile_model, strategy_scope
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.scripts.utils.initialization import check_all_paths_exist

# TODO: check
# <https://stackoverflow.com/a/58970598/5811400> and <https://github.com/googlecolab/colabtools/issues/864#issuecomment-556437040> # noqa
# TODO: na condensação, fazer crop determinístico!!!
# TODO: depois, se tiver tempo, fazer RandomCrop pra comparar
# TODO: ver o quanto influencia crop determinístico versus randomico
# TODO: para ambos os tipos de corte, rodar auto ML
# TODO: para um mesmo fluxo, mostrar imagens para os quatro datasets
# TODO: fazer vídeos tipo o do Hobold com a ebulição e um gráfico (barrinha de erro), o
# fluxo nominal e o valor predito
# TODO: esse vídeo pode ser para as quatro superfícies ao mesmo tempo

strategy = configure(
    force_gpu_allow_growth=True,
    use_xla=True,
    # mixed_precision_global_policy='mixed_float16',
    modin_engine='ray',
    require_gpu=True,
)

console = Console()

logger.info('Checking paths')
check_all_paths_exist(
    (
        ('Boiling cases', boiling_data_path()),
        ('Condensation cases', condensation_data_path()),
        ('Analyses', analyses_path()),
    )
)
logger.info('Succesfully checked paths')


# for is_direct, datasets in (
#     (True, BOILING_DIRECT_DATASETS),
#     (False, BOILING_INDIRECT_DATASETS),
# ):
#     for index, dataset in enumerate(datasets):
#         for subset_name, subset_ in zip(('train', 'val', 'test'), dataset()):
#             logger.info(
#                 f"Iterating over {'direct' if is_direct else 'indirect'} {subset_name} "
#                 f'dataset #{index}.'
#             )
#             for frame, targets_ in subset_:
#                 pass

# logger.debug("Done")


"""### Baseline on-wire pool boiling"""
logger.info('Getting direct baseline model')
baseline_boiling_model_architecture_direct = get_baseline_boiling_architecture(
    direct_visualization=True,
    normalize_images=False,
    strategy=strategy,
)
logger.info('Done getting direct baseline model')

logger.info('Getting indirect baseline model')
baseline_boiling_model_architecture_indirect = get_baseline_boiling_architecture(
    direct_visualization=False,
    normalize_images=False,
    strategy=strategy,
)
logger.info('Done getting indirect baseline model')

baseline_boiling_model_direct_size = int(
    baseline_boiling_model_architecture_direct.count_parameters(
        trainable=True,
        non_trainable=False,
    )
)

baseline_boiling_model_indirect_size = int(
    baseline_boiling_model_architecture_indirect.count_parameters(
        trainable=True,
        non_trainable=False,
    )
)

pretrained_baseline_boiling_model_architecture_direct = get_pretrained_baseline_boiling_model(
    direct_visualization=True,
    normalize_images=False,
    strategy=strategy,
)
pretrained_baseline_boiling_model_architecture_indirect = get_pretrained_baseline_boiling_model(
    direct_visualization=False,
    normalize_images=False,
    strategy=strategy,
)


"""## Pre-processing analyses"""

# TODO: generate a test case in which temperature is estimated from the boiling curve and that's
# what the models have to predict

# TODO: data clean up; remove data from experiments where measured heatflux is 5W/cm2 or more away
# from its level
# TODO: fazer erro em função do y: ver se para maiores ys o erro vai subindo ou diminuindo
# quem sabe fazer 3 ou mais modelos, um especializado para cada região de y; e quem sabe
# usar um classificador pra escolher qual estimador não ajude muito
# focar na arquitetura da rede, que é mais importante do que hiperparâmetros
# otimizar as convolucionais pode ser mais importante do que otimizar as fully-connected

# TODO: hypothesis: it is better to _not_ normalize images.
# Does this mean that the model uses the overall image brightness to do its inference?
# If so, this is bad.
# This can be assessed by training two models (one with normalized images and the other without)
# and comparing the relative importance that each one of them gives to the areas without bubbles

# TODO: does the model tend to overestimate or underestimate values?
# TODO: train the same network multiple times to get an average and stddev of the error. I noticed
# that, by training the same model multiple times, I got R2 scores of ~0.96, 0.90 and 0.94.
# Hobold got 0.98... Maybe it's just because he tried a lot of times until he got a good
# performance?

"""### Data Distribution"""


PREFETCH = 1024 * 4


# @cache(JSONAllocator(analyses_path() / 'cache' / 'targets'))
# def get_targets(
#     dataset: LazyDescribed[ImageDatasetTriplet],
# ) -> tuple[list[Targets], list[Targets], list[Targets]]:
#     ds_train, ds_val, ds_test = dataset()

#     return (
#         list(targets(ds_train).prefetch(PREFETCH)),
#         list(targets(ds_val).prefetch(PREFETCH)),
#         list(targets(ds_test).prefetch(PREFETCH)),
#     )


# def plot_dataset_targets(
#     datasets: tuple[LazyDescribed[ImageDatasetTriplet], ...],
#     *,
#     target_name: str,
#     filter_target: Optional[Callable[[Targets], bool]] = None,
# ) -> None:
#     sns.set_style('whitegrid')

#     f, axes = plt.subplots(len(datasets), 3, figsize=(9, 9), sharey='row')

#     for row, splits in enumerate(datasets):
#         targets_train, targets_val, targets_test = get_targets(splits)
#         for col, (title, ys) in enumerate(
#             (
#                 ('train', targets_train),
#                 ('val', targets_val),
#                 ('test', targets_test),
#             )
#         ):
#             y = [
#                 target[target_name]
#                 for target in ys
#                 if filter_target is None or filter_target(target)
#             ]
#             x = range(len(y))
#             axes[row, col].scatter(x, y)

#             if not row:
#                 axes[row, col].set_title(title)

#             if not col:
#                 axes[row, col].set_ylabel(f'Dataset #{row + 1}')


"""#### On-wire pool boiling"""


# plot_dataset_targets(
#     BOILING_DIRECT_DATASETS,
#     target_name=DEFAULT_BOILING_HEAT_FLUX_TARGET,
#     filter_target=BOILING_OUTLIER_FILTER
# )

# plot_dataset_targets(
#     BOILING_INDIRECT_DATASETS,
#     target_name=DEFAULT_BOILING_HEAT_FLUX_TARGET,
#     filter_target=BOILING_OUTLIER_FILTER
# )

"""#### Condensation"""

# TODO: ensure that this works!
# plot_dataset_targets(CONDENSATION_DATASETS())

"""### Downscaling"""


"""### Consecutive frames"""

# TODO: fix this!!!
# since the dataset is already shuffled, I'm not taking consecutive frames

# TODO: test defining the hold-out sets as literal slices, as in:
# ds_train, ds_val, ds_test = ds[:X], ds[X:Y], ds[Y:]
# where ds is NOT shuffled


# frames_indices = tuple(range(10)) + tuple(range(10, 100, 10))

# metrics = [
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio,
#     normalized_mutual_information
# ]

# sns.set_style("whitegrid")

# f, axes = plt.subplots(
#     len(metrics), len(BOILING_DIRECT_DATASETS), figsize=(16, 16), sharex="row", sharey="col"
# )

# x = [index + 1 for index in frames_indices]
# for col, splits in enumerate(BOILING_DIRECT_DATASETS):
#     ds_train, _, _ = splits()
#     frames = features(ds_train).fetch(frames_indices)
#     for row, metric in enumerate(metrics):
#         ax = axes[row, col]

#         y = [metric(frames[0], frame) for frame in frames]

#         ax.scatter(x, y, s=20, color='k')
#         ax.scatter(x[0], y[0], facecolors="none", edgecolors="k", marker="$\odot$", s=100)

#         if not row:
#             ax.set_title(f"Dataset {col}")
#         if not col:
#             ax.set_ylabel(" ".join(metric.__name__.split("_")).title())

#         ax.xaxis.grid(True, which='minor')
#         ax.set_xscale("log")


# f, axes = plt.subplots(
#     len(BOILING_DIRECT_DATASETS), 3, figsize=(10, 16), sharex="row", sharey="col"
# )

# x = [index + 1 for index in frames_indices]
# for row, splits in enumerate(BOILING_DIRECT_DATASETS):
#     for col, split_name, split in zip(range(3), ("Train", "Val", "Test"), splits()):
#         ax = axes[row, col]
#         frame, data = split.shuffle()[0]

#         if not row:
#             ax.set_title(split_name)
#         if not col:
#             ax.set_ylabel(f"Dataset {row + 1}")

#         ax.set_xlabel(f"{data[DEFAULT_BOILING_HEAT_FLUX_TARGET]:.2f}W/cm² (#{data['index']})")
#         ax.imshow(frame.squeeze(), cmap="gray")
#         ax.grid(False)

"""### On-Wire Pool Boiling

#### Validation
"""
baseline_boiling_dataset_direct = boiling_datasets(direct_visualization=True)[0]
baseline_boiling_dataset_indirect = boiling_datasets(direct_visualization=False)[0]


# !pip install shap

# import shap

# import numpy as np

# background = np.array(features(baseline_boiling_dataset_direct()[0].sample(100)))
# samples = np.array(features(baseline_boiling_dataset_direct()[1].sample(4)))

# e = shap.DeepExplainer(validated_model_direct.architecture.model, background)

# shap_values = e.shap_values(samples, check_additivity=False)

# shap.image_plot(shap_values, samples)

# print(baseline_boiling_dataset_direct()[0][0][0].shape)
# print(baseline_boiling_dataset_indirect()[0][0][0].shape)

# import numpy as np

# background = np.array(features(baseline_boiling_dataset_indirect()[0].sample(100)))
# samples = np.array(features(baseline_boiling_dataset_indirect()[1].sample(4)))

# e = shap.DeepExplainer(validated_model_indirect.architecture.model, background)

# shap_values = e.shap_values(samples, check_additivity=False)

# shap.image_plot(shap_values, samples)

# # explain how the input to the 7th layer of the model explains the top two classes
# def map2layer(x, layer):
#     feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
#     return K.get_session().run(model.layers[layer].input, feed_dict)

# e = shap.GradientExplainer(
#     (model.layers[7].input, model.layers[-1].output),
#     map2layer(X, 7),
#     local_smoothing=0 # std dev of smoothing noise
# )
# shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)

# # get the names for the classes
# index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# # plot the explanations
# shap.image_plot(shap_values, to_explain, index_names)

# !pip install tf-explain

# from tf_explain.core.grad_cam import GradCAM

# explainer = GradCAM()
# grid = explainer.explain(baseline_boiling_dataset_direct()[1][0], model.architecture.model, )


"""#### Cross-surface boiling evaluation with AutoML"""

# %tensorboard --logdir $tensorboard_logs_path


# logger.info('Analyzing cross-surface boiling evaluation with AutoML')
# BOILING_BASELINE_BATCH_SIZE = 200


# @cache(JSONTableAllocator(analyses_path() / 'studies' / 'boiling-cross-surface-automl'))
# def boiling_cross_surface_evaluation_automl(
#     direct_visualization: bool, training_cases: tuple[int, ...], evaluation_cases: tuple[int, ...]
# ) -> dict[str, float]:
#     logger.info(
#         f'Training on cases {training_cases} '
#         f'| evaluation on {evaluation_cases} '
#         f"| {'Direct' if direct_visualization else 'Indirect'} visualization"
#     )

#     all_datasets = BOILING_DIRECT_DATASETS if direct_visualization else BOILING_INDIRECT_DATASETS
#     training_datasets = tuple(all_datasets[training_case] for training_case in training_cases)
#     evaluation_datasets = tuple(
#         all_datasets[evaluation_case] for evaluation_case in evaluation_cases
#     )

#     training_dataset = LazyDescribed.from_describable(training_datasets) | datasets_merger()
#     evaluation_dataset = LazyDescribed.from_describable(evaluation_datasets) | datasets_merger()

#     tune_model_return: TuneModelReturn = autofit_to_dataset(
#         training_dataset,
#         target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
#         normalize_images=True,
#         max_model_size=baseline_boiling_model_direct_size,
#         goal=None,
#     )

#     logger.info('Evaluating')
#     with strategy_scope(strategy):
#         compile_model(tune_model_return.model, get_baseline_compile_params())

#     ds_evaluation_val = to_tensorflow(
#         evaluation_dataset | subset('val'),
#         batch_size=BOILING_BASELINE_BATCH_SIZE,
#         target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
#     )

#     evaluation = tune_model_return.model.evaluate(ds_evaluation_val())
#     logger.info(f'Done: {evaluation}')

#     return evaluation


# cases_indices = ((0,), (1,), (0, 1), (2,), (3,), (2, 3), (0, 1, 2, 3))

# boiling_cross_surface_automl = {
#     (is_direct, training_cases, evaluation_cases): boiling_cross_surface_evaluation_automl(
#         is_direct, training_cases, evaluation_cases
#     )
#     for is_direct, training_cases, evaluation_cases in itertools.product(
#         (False, True), cases_indices, cases_indices
#     )
# }

"""#### First Condensation Case"""

# TODO: interesting analysis:
# since the parametric studies use the same type of surface, I would expect that the network would
# get more confused

logger.info('First condensation case')


def _set_case_name(data: Targets) -> Targets:
    # TODO: this should be set by `set_condensation_datasets_data`
    data['case_name'] = ':'.join(data['name'].split(':')[:2])
    return data


logger.info('Getting datasets...')
condensation_all_cases = ExperimentVideoDataset().union(*condensation_datasets())
ds = condensation_dataset(each=60)
ds_train, ds_val, ds_test = ds()
ds_train = map_targets(ds_train, _set_case_name)
ds_val = map_targets(ds_val, _set_case_name)
ds_test = map_targets(ds_test, _set_case_name)
logger.info('Done')

logger.info('Calculating classes')
CLASSES = sorted(frozenset(targets(ds_train).map(itemgetter('case_name')).prefetch(4096)))
N_CLASSES = len(CLASSES)


def _set_case(data: Targets) -> Targets:
    data['case'] = CLASSES.index(data['case_name'])
    return data


ds_train = map_targets(ds_train, _set_case)
ds_val = map_targets(ds_val, _set_case)
ds_test = map_targets(ds_test, _set_case)
logger.info('Done')

logger.info('Describing datasets...')
datasets = LazyDescribed.from_value_and_description(
    value=(ds_train, ds_val, ds_test), description=ds
)
logger.info('Done')

logger.info('Getting first frame...')
first_frame, _ = ds_train[0]
logger.info('Done')

logger.info('Compiling...')
with strategy_scope(strategy):
    architecture = hoboldnet2(
        first_frame.shape,
        dropout=0.5,
        output_layer_policy='float32',
        problem='classification',
        num_classes=N_CLASSES,
    )
    compile_params = CompileModelParams(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name='top3'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5'),
            # tfa.metrics.F1Score(N_CLASSES, name='F1'),
        ],
    )
    compiled_model = compile_model(architecture, compile_params)

assert False, 'STOP!'


# """### FPS analysis"""

# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline


# # TODO: is FPS 30 or 1 here???
# # Answer: take a look at
# <https://drive.google.com/drive/u/1/folders/1hVLDeLOlklVqIN-W6eGbRUqTxMXZTF74>
# # It seems that FPS is already 1, so frame #30 happens 30s after frame #0.

# for reference_datasets, preprocessors, final_timeshift, timeshifts in (
#     (
#         reference_datasets_boiling,
#         boiling_direct_preprocessors,
#         1,
#         (0, 1, 2, 3, 5, 10, 20, 30, 60),
#     ),
#     (
#         reference_datasets_condensation,
#         condensation_preprocessors,
#         300,
#         (0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1600, 3600, 7200),
#     ),
# ):
#     for dataset in reference_datasets:
#         n_frames = max(timeshifts) + 1

#         preprocessors = select_preprocessors(preprocessors)

#         dataset = bl.datasets.apply_transformers(dataset, preprocessors)
#         _, data = list(dataset.take(1).as_numpy_iterator())[0]
#         print(data)
#         frames = {
#             idx: frame
#             for idx, frame in enumerate(
#                 dataset.map(lambda image, data: image).take(n_frames).as_numpy_iterator()
#             )
#             if idx in timeshifts
#         }

#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         ax.imshow(frames[0], cmap='gray')
#         fig.show()

#         analyze_consecutive_frames.main(
#             frames.items(),
#             metrics={
#                 'Retained variance': retained_variance,
#                 'Cross-entropy ratio': shannon_cross_entropy_ratio,
#                 'Entropy ratio': shannon_entropy_ratio,
#                 'NMI ratio': normalized_mutual_information,
#                 'Structural similarity': structural_similarity_ratio,
#             },
#             timeshifts=timeshifts,
#             final_timeshift=final_timeshift,
#             xscale='symlog',
#             figsize=(4, 3),
#         )


# """## TO-DO List

# - [ ] improve sectioning
# - [ ] remove old code
# - [ ] remove unnecessary comments
# - [ ] make "Boiling Learning.ipynb" deprecated
# - [ ] see available models at https://github.com/tensorflow/models/tree/master/official
# - [ ] see more models at https://tfhub.dev/tensorflow/efficientnet/b0/classification/1
# - [ ] see TensorFlow Addons, such as the following optimizer:
#     https://www.tensorflow.org/addons/tutorials/optimizers_lazyadam
# - [ ] include
#     https://www.tensorflow.org/addons/tutorials/tqdm_progress_bar#default_tqdmcallback_usage
# - [x] change `Manager`s path from `test_` to real
# - [x] improve shuffling... we can see that data is almost contiguous
# - [x] calculate heat flux and use it instead of power
# - [ ] see papers in Kopernio
# - [x] when describing a dataset, include only the DictImageTransformer components that appear as
#     ExperimentVideos
# - [ ] allow bootstraping for confidence intervals in `bl.model.eval_with`
# - [ ] define function `eval_regions` that allows us to evaluate a model when
#     nominal power = 0, 10, 20... for instance:
#     `eval_regions(model, ds_val, 'nominal_heat_flux') == {0: 8.32, 10: 5.63, 20: 12.10, ...}`
# """
