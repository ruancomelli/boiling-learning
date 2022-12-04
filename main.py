import itertools
from fractions import Fraction
from operator import itemgetter
from pprint import pprint

import more_itertools as mit
import tensorflow as tf
from loguru import logger
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import BOILING_BASELINE_BATCH_SIZE
from boiling_learning.app.datasets.bridging import to_tensorflow
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.app.datasets.raw.boiling1d import boiling_data_path
from boiling_learning.app.datasets.raw.condensation import (
    condensation_data_path,
    condensation_datasets,
)
from boiling_learning.app.paths import analyses_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_baseline_boiling_architecture,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.condensation import fit_condensation_model
from boiling_learning.datasets.sliceable import map_targets, targets
from boiling_learning.image_datasets import Targets
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.training import (
    CompileModelParams,
    FitModelParams,
    compile_model,
    strategy_scope,
)
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset
from boiling_learning.scripts.utils.initialization import check_all_paths_exist
from boiling_learning.transforms import dataset_sampler, datasets_merger, subset

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
    mixed_precision_global_policy='mixed_float16',
    modin_engine='ray',
    require_gpu=True,
)


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


"""#### Condensation"""

condensation_dataset_train, _, _ = condensation_dataset()()
print(mit.ilen(condensation_dataset_train[::60].prefetch(128 * 2)))


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


(
    condensation_dataset_train,
    _condensation_dataset_val,
    _condensation_dataset_test,
) = condensation_dataset()()

first_frame, _data = condensation_dataset_train[0]
with strategy_scope(strategy):
    architecture = hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=True)

compiled_model = compile_model(
    architecture,
    get_baseline_compile_params(strategy=strategy),
)
trained_model = fit_condensation_model(
    compiled_model,
    condensation_dataset(),
    get_baseline_fit_params(),
    target='mass_rate',
    strategy=strategy,
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

# from itertools import takewhile
#

# from boiling_learning.preprocessing.image import (
#     Downscaler,
#     normalized_mutual_information,
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio
# )

# preprocessors = list(
#     takewhile(
#         lambda preprocessor: not isinstance(preprocessor, Downscaler),
#         boiling_direct_preprocessors
#     )
# )

# sample_frames: list[Image] = []
# for case in boiling_cases():
#     get_image_dataset_params = GetImageDatasetParams(
#         case,
#         transformers=preprocessors,
#     )

#     logger.info(f"Getting datasets...")
#     ds_train, _, _ = get_image_dataset(get_image_dataset_params)()
#     logger.info(f"Done")

#     sample_frame, _ = ds_train[0]
#     sample_frames.append(sample_frame)

# import matplotlib.ticker as tck

# from skimage.metrics import normalized_mutual_information as nmi

# factors = range(1, 10)

# metrics = [
#     retained_variance,
#     shannon_cross_entropy_ratio,
#     shannon_entropy_ratio,
#     structural_similarity_ratio,
#     normalized_mutual_information,
#     nmi
# ]

# sns.set_style("whitegrid")

# f, axes = plt.subplots(
#     len(metrics), len(sample_frames), figsize=(16, 16), sharex="row", sharey="col"
# )

# x = factors
# preferred_factor = 4
# for col, sample_frame in enumerate(sample_frames):
#     downscaled_frames = [Downscaler(factor)(sample_frame) for factor in factors]

#     for row, metric in enumerate(metrics):
#         ax = axes[row, col]

#         y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

#         ax.scatter(x, y, s=20, color='k')
#         ax.scatter(x[0], y[0], facecolors="none", edgecolors="k", marker="$\odot$", s=100)
#         ax.scatter(
#             x[preferred_factor],
#             y[preferred_factor],
#             facecolors="none",
#             edgecolors="k",
#             marker="$\odot$",
#             s=100
#         )

#         if not row:
#             ax.set_title(f"Dataset {col}")
#         if not col:
#             ax.set_ylabel(" ".join(metric.__name__.split("_")).title())

#         ax.xaxis.grid(True, which='minor')

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

"""FIRST CONDENSATION CASE JUST FOR FUN"""

(
    condensation_dataset_train,
    _condensation_dataset_val,
    _condensation_dataset_test,
) = condensation_dataset()

first_frame, _data = condensation_dataset_train[0]
with strategy_scope(strategy):
    architecture = hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=True)

compiled_model = compile_model(
    architecture,
    get_baseline_compile_params(),
)
trained_model = fit_condensation_model(
    compiled_model,
    condensation_dataset,
    get_baseline_fit_params(),
    target='mass_rate',
)

"""### On-Wire Pool Boiling

#### Validation
"""
baseline_boiling_dataset_direct = boiling_datasets(direct_visualization=True)[0]
baseline_boiling_dataset_indirect = boiling_datasets(direct_visualization=False)[0]

validated_model_direct = get_pretrained_baseline_boiling_model(
    direct_visualization=True,
    normalize_images=False,
    strategy=strategy,
)
print(validated_model_direct)
print('Evaluation:', validated_model_direct.evaluation)

validated_model_indirect = get_pretrained_baseline_boiling_model(
    direct_visualization=False,
    normalize_images=False,
    strategy=strategy,
)
print(validated_model_indirect)
print('Evaluation:', validated_model_indirect.evaluation)

validated_model_direct_normalized = get_pretrained_baseline_boiling_model(
    direct_visualization=True,
    normalize_images=True,
    strategy=strategy,
)
print(validated_model_direct_normalized)
print('Evaluation:', validated_model_direct_normalized.evaluation)

validated_model_indirect_normalized = get_pretrained_baseline_boiling_model(
    direct_visualization=False,
    normalize_images=True,
    strategy=strategy,
)
print(validated_model_indirect_normalized)
print('Evaluation:', validated_model_indirect_normalized.evaluation)

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

"""#### Retrain randomness"""

logger.info('Analyzing effects of random initialization')

NUMBER_OF_RETRAINS = 8
evaluations = []
for retrain_index in range(NUMBER_OF_RETRAINS):
    compiled_model = compile_model(
        get_baseline_boiling_architecture(
            direct_visualization=True,
            normalize_images=False,
            strategy=strategy,
        ),
        get_baseline_compile_params(strategy=strategy),
    )

    model = fit_boiling_model(
        compiled_model,
        baseline_boiling_dataset_direct,
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
        try_id=retrain_index,
    )

    print(model)
    best_performance = min(model.history, key=lambda data: data['val_loss'])
    evaluations.append(best_performance)
    logger.info('Done')

pprint(evaluations)

"""#### Other wire - training"""

# %%time
# %tensorboard --logdir $tensorboard_logs_path


logger.info('Testing with other wire')

DATASET = boiling_datasets(direct_visualization=True)[2]

logger.info('Compiling...')
first_frame, _ = DATASET()[0][0]

with strategy_scope(strategy):
    # TODO: replace this with utility function!
    architecture = hoboldnet2(first_frame.shape, dropout=0.5, normalize_images=False)
    compiled_model = compile_model(architecture, get_baseline_compile_params(strategy=strategy))
logger.info('Done')

logger.info('Training...')
fit_model_params = FitModelParams(
    batch_size=200,
    epochs=100,
    callbacks=LazyDescribed.from_list(
        [
            LazyDescribed.from_constructor(tf.keras.callbacks.TerminateOnNaN),
            LazyDescribed.from_constructor(
                tf.keras.callbacks.EarlyStopping,
                monitor='val_loss',
                min_delta=0,
                patience=10,
                baseline=None,
                mode='auto',
                restore_best_weights=True,
                verbose=1,
            ),
        ]
    ),
)

with strategy_scope(strategy):
    model = fit_boiling_model(
        compiled_model,
        DATASET,
        fit_model_params,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
    )
pprint(model)
logger.info('Done')

"""#### Boiling learning curve"""

# %%time
# %tensorboard --logdir $tensorboard_logs_path


logger.info('Analyzing learning curve')

ds_evaluation_direct = to_tensorflow(
    baseline_boiling_dataset_direct | subset('val'),
    batch_size=BOILING_BASELINE_BATCH_SIZE,
    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    experiment='boiling1d',
)

ds_evaluation_indirect = to_tensorflow(
    baseline_boiling_dataset_indirect | subset('val'),
    batch_size=BOILING_BASELINE_BATCH_SIZE,
    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    experiment='boiling1d',
)


@cache(JSONAllocator(analyses_path() / 'studies' / 'boiling-learning-curve'))
def boiling_learning_curve_point(
    fraction: Fraction, *, direct_visualization: bool = True, normalize_images: bool = False
) -> dict[str, float]:
    logger.info(f'Analyzing fraction {fraction}')

    logger.info('Getting datasets...')
    datasets = (
        baseline_boiling_dataset_direct
        if direct_visualization
        else baseline_boiling_dataset_indirect
    ) | dataset_sampler(fraction)
    logger.info('Done')

    logger.info('Compiling...')
    with strategy_scope(strategy):
        compiled_model = compile_model(
            get_baseline_boiling_architecture(
                direct_visualization=direct_visualization,
                normalize_images=normalize_images,
                strategy=strategy,
            ),
            get_baseline_compile_params(strategy=strategy),
        )
    logger.info('Done')

    logger.info('Training...')

    model = fit_boiling_model(
        compiled_model,
        datasets,
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    )

    logger.info(f'Evaluating: {fraction}')
    with strategy_scope(strategy):
        compile_model(model.architecture, get_baseline_compile_params())
    evaluation = model.architecture.evaluate(
        (ds_evaluation_direct if direct_visualization else ds_evaluation_indirect)()
    )

    logger.info('Done')

    return evaluation


boiling_learning_curve = {
    fraction: boiling_learning_curve_point(
        fraction, direct_visualization=direct, normalize_images=normalize_images
    )
    for fraction in (
        Fraction(1, 100),
        Fraction(1, 20),
        Fraction(1, 10),
        Fraction(1, 2),
        Fraction(1, 1),
    )
    for direct in (False, True)
    for normalize_images in (False, True)
}


METRIC_NAMES = ('MSE', 'RMS', 'MAE', 'MAPE', 'R2')

console = Console()

learning_curve_analysis = Table(title='Learning curve analysis')
learning_curve_analysis.add_column('Dataset fraction', justify='center')

for metric_name in METRIC_NAMES:
    learning_curve_analysis.add_column(metric_name, justify='right')

for fraction, metrics in boiling_learning_curve.items():
    learning_curve_analysis.add_row(
        str(fraction), *(f'{metrics[metric_name]:.2f}' for metric_name in METRIC_NAMES)
    )

console.print(learning_curve_analysis)

"""#### Auto ML"""

regular_wire_best_model_direct_visualization = autofit_dataset(
    baseline_boiling_dataset_direct,
    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    normalize_images=True,
    max_model_size=baseline_boiling_model_direct_size,
    goal=None,
    experiment='boiling1d',
)

print(regular_wire_best_model_direct_visualization)

regular_wire_best_model_indirect_visualization = autofit_dataset(
    baseline_boiling_dataset_indirect,
    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    normalize_images=True,
    max_model_size=baseline_boiling_model_indirect_size,
    goal=None,
    experiment='boiling1d',
)

print(regular_wire_best_model_indirect_visualization)

"""#### AutoML - Less data"""


ds_train = baseline_boiling_dataset_direct | dataset_sampler(Fraction(1, 100)) | subset('train')
ds_val = baseline_boiling_dataset_direct | subset('val')

regular_wire_best_model_direct_visualization_less_data = autofit_dataset(
    LazyDescribed.from_value_and_description(
        (ds_train(), ds_val(), None), (ds_train, ds_val, None)
    ),
    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    normalize_images=True,
    max_model_size=baseline_boiling_model_indirect_size,
    goal=None,
    experiment='boiling1d',
)

print(regular_wire_best_model_direct_visualization_less_data)

"""#### Other wire - auto ML"""


# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# hypermodel = ConvImageRegressor(
#     loss=loss,
#     metrics=metrics,
#     tuner=EarlyStoppingGreedy,
#     directory=hypermodel_allocator,
#     max_model_size=int(
#         baseline_boiling_model_architecture.count_parameters(trainable=True, non_trainable=False)
#     ),
#     strategy=strategy,
#     goal=baseline_boiling_loss,
#     normalize_images=False,
# )

# tune_model_params = TuneModelParams(
#     batch_size=16,
#     callbacks=Described.from_list(
#         [
#             Described.from_constructor(tf.keras.callbacks.TerminateOnNaN, P()),
#             Described.from_constructor(
#                 tf.keras.callbacks.EarlyStopping,
#                 P(
#                     monitor='val_loss',
#                     min_delta=0,
#                     # patience=2,
#                     patience=10,
#                     baseline=None,
#                     mode='auto',
#                     restore_best_weights=True,
#                     verbose=1,
#                 ),
#             ),
#         ]
#     ),
# )

# regular_wire_best_model = autofit(
#     hypermodel,
#     datasets=BOILING_DIRECT_DATASETS[1],
#     params=tune_model_params,
#     target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
# )
# print(regular_wire_best_model)


logger.info('Analyzing cross-surface boiling evaluation')


@cache(JSONAllocator(analyses_path() / 'studies' / 'boiling-cross-surface'))
def boiling_cross_surface_evaluation(
    direct_visualization: bool,
    training_cases: tuple[int, ...],
    evaluation_cases: tuple[int, ...],
    normalize_images: bool = True,
) -> dict[str, float]:
    logger.info(
        f'Training on cases {training_cases} '
        f'| evaluation on {evaluation_cases} '
        f"| {'Direct' if direct_visualization else 'Indirect'} visualization"
    )

    all_datasets = boiling_datasets(direct_visualization=direct_visualization)
    training_datasets = tuple(all_datasets[training_case] for training_case in training_cases)
    evaluation_datasets = tuple(
        all_datasets[evaluation_case] for evaluation_case in evaluation_cases
    )

    training_dataset = LazyDescribed.from_describable(training_datasets) | datasets_merger()
    evaluation_dataset = LazyDescribed.from_describable(evaluation_datasets) | datasets_merger()

    with strategy_scope(strategy):
        architecture = get_baseline_boiling_architecture(
            direct_visualization=direct_visualization,
            normalize_images=normalize_images,
        )
        compiled_model = compile_model(architecture, get_baseline_compile_params())

    logger.info('Training...')

    model = fit_boiling_model(
        compiled_model,
        training_dataset,
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
    )

    logger.info('Evaluating')
    with strategy_scope(strategy):
        compile_model(model.architecture, get_baseline_compile_params())

    ds_evaluation_val = to_tensorflow(
        evaluation_dataset | subset('val'),
        batch_size=BOILING_BASELINE_BATCH_SIZE,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        experiment='boiling1d',
    )

    print(next(ds_evaluation_val().as_numpy_iterator()))

    evaluation = model.architecture.evaluate(ds_evaluation_val())
    logger.info(f'Done: {evaluation}')

    return evaluation


cases_indices = ((0,), (1,), (0, 1), (2,), (3,), (2, 3), (0, 1, 2, 3))

boiling_cross_surface = {
    (is_direct, training_cases, evaluation_cases): boiling_cross_surface_evaluation(
        is_direct, training_cases, evaluation_cases, normalize_images=True
    )
    for is_direct, training_cases, evaluation_cases in itertools.product(
        (False, True), cases_indices, cases_indices
    )
}

print(boiling_cross_surface)


console = Console()


def _format_sets(indices: tuple[int, ...]) -> str:
    return ' + '.join(map(str, indices))


def _get_and_format_results(direct_result: float, indirect_result: float) -> str:
    formatted_direct_result = f'[bold]{direct_result:.4f}[/bold]'
    formatted_indirect_result = f'{indirect_result:.4f}'

    ratio = (indirect_result - direct_result) / direct_result
    formatted_ratio = (
        f'[bold][bright_red]{ratio:+.2%}[/bright_red][/bold]'
        if ratio > 0
        else f'[bold][bright_green]{ratio:+.2%}[/bright_green][/bold]'
    )

    return f'{formatted_direct_result}\n{formatted_indirect_result}\n({formatted_ratio})'


for metric_name in ('MSE', 'MAPE', 'RMS', 'R2'):
    cross_surface_analysis = Table(
        'Train \\ Eval',
        *(map(_format_sets, cases_indices)),
        title=f'Cross surface analysis - {metric_name}',
    )

    for training_indices in cases_indices:
        cross_surface_analysis.add_row(
            _format_sets(training_indices),
            *map(
                _get_and_format_results,
                (
                    boiling_cross_surface[(True, training_indices, evaluation_cases)][metric_name]
                    for evaluation_cases in cases_indices
                ),
                (
                    boiling_cross_surface[(False, training_indices, evaluation_cases)][metric_name]
                    for evaluation_cases in cases_indices
                ),
            ),
            end_section=True,
        )

    console.print(cross_surface_analysis)

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
ds = condensation_dataset()
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

# """## The End

# ## Experimental Code

# ### Kramer data
# """


# """### AutoKeras"""


# BOILING_BASELINE_BATCH_SIZE = 32

# get_image_dataset_params = GetImageDatasetParams(
#     boiling_cases()[0],
#     transformers=(*boiling_direct_preprocessors, ImageNormalizer()),
#     dataset_size=None,
# )

# logger.info('Getting datasets...')
# datasets = Described(get_image_dataset(get_image_dataset_params), get_image_dataset_params)
# ds_train, ds_val, ds_test = datasets.value
# first_frame, _ = ds_train[0]
# ds_train, ds_val, _ = to_tensorflow_triplet(
#     datasets,
#     batch_size=BOILING_BASELINE_BATCH_SIZE,
#     include_test=False,
#     target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
# )
# ds_train = ds_train.unbatch().prefetch(tf.data.AUTOTUNE)
# ds_val = ds_val.unbatch().prefetch(tf.data.AUTOTUNE)
# logger.info('Done')

# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# inputs = ak.ImageInput()
# x = LayersBlock(baseline_boiling_model_architecture.model.layers[1:-3])(inputs)
# outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(x)

# regressor = ak.AutoModel(
#     inputs,
#     outputs,
#     distribution_strategy=strategy.value,
#     tuner=EarlyStoppingGreedy,
#     overwrite=True,
#     goal=10,
#     # max_model_size=sum(
#     #     tf.keras.backend.count_params(p)
#     #     for p in baseline_boiling_model_architecture.model.trainable_weights
#     # )
# )


# regressor.fit(
#     ds_train,
#     validation_data=ds_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.TerminateOnNaN(),
#         # TimePrinter()
#     ],
#     batch_size=BOILING_BASELINE_BATCH_SIZE,
# )


# logger.info('Done')


# BOILING_BASELINE_BATCH_SIZE = 32

# get_image_dataset_params = GetImageDatasetParams(
#     boiling_cases()[0],
#     transformers=(*boiling_direct_preprocessors, ImageNormalizer()),
#     dataset_size=None,
# )

# logger.info('Getting datasets...')
# datasets = Described(get_image_dataset(get_image_dataset_params), get_image_dataset_params)
# ds_train, ds_val, ds_test = datasets.value
# first_frame, _ = ds_train[0]
# ds_train, ds_val, _ = to_tensorflow_triplet(
#     datasets,
#     batch_size=BOILING_BASELINE_BATCH_SIZE,
#     include_test=False,
#     target=DEFAULT_BOILING_HEAT_FLUX_TARGET
# )
# ds_train = ds_train.unbatch().prefetch(tf.data.AUTOTUNE)
# ds_val = ds_val.unbatch().prefetch(tf.data.AUTOTUNE)
# logger.info('Done')

# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# inputs = ak.ImageInput()
# x = ak.ImageBlock(normalize=False, augment=False)(inputs)
# x = ak.SpatialReduction()(x)
# x = ak.DenseBlock()(x)
# outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(x)

# regressor = ak.AutoModel(
#     inputs, outputs, distribution_strategy=strategy.value, tuner=BetterGreedy, overwrite=True
# )

# regressor.fit(
#     ds_train,
#     validation_data=ds_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.TerminateOnNaN(),
#         # TimePrinter()
#     ],
#     batch_size=BOILING_BASELINE_BATCH_SIZE,
# )


# logger.info('Done')


# save_best_epoch_on_epoch_end = SaveBestEpoch.on_epoch_end
# SaveBestEpoch.on_epoch_end = Callback.on_epoch_end


# class CNN2Block(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Conv2D(32, (5, 5), padding='same')(input_node)
#         x = ReLU()(x)
#         x = MaxPool2D((2, 2), strides=(2, 2))(x)
#         x = Dropout(0.5)(x)
#         # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         return x


# class DenseBlock(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Dense(200)(input_node)
#         x = ReLU()(x)
#         x = Dropout(0.5)(x)
#         # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         x = Dense(1)(x)
#         return Activation('linear')(x)


# class HoboldNetBlock(ak.Block):
#     def build(self, hp, inputs=None):
#         # Get the input_node from inputs.
#         input_node = tf.nest.flatten(inputs)[0]
#         x = Conv2D(32, (5, 5), padding='same', dtype='mixed_float16')(input_node)
#         x = ReLU(dtype='mixed_float16')(x)
#         x = MaxPool2D((2, 2), strides=(2, 2), dtype='mixed_float16')(x)
#         x = Flatten(dtype='mixed_float16')(x)
#         x = Dropout(0.5, dtype='mixed_float16')(x)
#         x = Dense(200, dtype='mixed_float16')(x)
#         x = ReLU(dtype='mixed_float16')(x)
#         return x
#         # x = Dropout(0.5)(x)
#         # # x = Dropout(hp.Float("dropout", min_value=0.0, max_value=1.0))(x)
#         # x = Dense(1)(x)
#         # return Activation('linear')(x)


# with strategy_scope(strategy):
#     loss = tf.keras.losses.MeanSquaredError()
#     metrics = [
#         tf.keras.metrics.MeanSquaredError('MSE'),
#         tf.keras.metrics.RootMeanSquaredError('RMS'),
#         tf.keras.metrics.MeanAbsoluteError('MAE'),
#         tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
#         tfa.metrics.RSquare('R2'),
#     ]

# input_node = ak.ImageInput()
# # x = ak.Normalization()(input_node)
# # x = CNN2Block()(input_node)
# # x = ak.SpatialReduction()(x)
# # x = DenseBlock()(x)
# # output_node = ak.RegressionHead()(x)
# # x = HoboldNetBlock()(input_node)
# # output_node = DenseBlock()(x)
# x = HoboldNetBlock()(input_node)
# output_node = ak.RegressionHead(loss=loss, dropout=0.5, output_dim=1, metrics=metrics)(x)

# auto_model = ak.AutoModel(
#     inputs=input_node,
#     outputs=output_node,
#     directory=analyses_path() / 'temp' / 'auto_tune_hoboldnet2-5',
#     overwrite=True,
#     distribution_strategy=strategy.value,
# )
# auto_model.fit(
#     ds_train.batch(
#         200,
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=False,
#     ).prefetch(tf.data.AUTOTUNE),
#     validation_data=ds_val.batch(
#         200,
#         num_parallel_calls=tf.data.AUTOTUNE,
#         deterministic=False,
#     ).prefetch(tf.data.AUTOTUNE),
#     # batch_size=200,
#     callbacks=[
#         tf.keras.callbacks.TerminateOnNaN(),
#         TimePrinter(
#             when={
#                 # 'on_batch_begin',
#                 # 'on_batch_end',
#                 'on_epoch_begin',
#                 'on_epoch_end',
#                 # 'on_predict_batch_begin',
#                 # 'on_predict_batch_end',
#                 # 'on_predict_begin',
#                 # 'on_predict_end',
#                 # 'on_test_batch_begin',
#                 # 'on_test_batch_end',
#                 # 'on_test_begin',
#                 # 'on_test_end',
#                 # 'on_train_batch_begin',
#                 # 'on_train_batch_end',
#                 # 'on_train_begin',
#                 # 'on_train_end',
#             }
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             min_delta=0,
#             patience=2,
#             # patience=10,
#             baseline=None,
#             mode='auto',
#             restore_best_weights=True,
#             verbose=1,
#         ),
#     ],
# )

# """## Old Code"""

# reference_datasets_boiling = (
#     boiling_cases()[0]['GOPR2868'].as_tf_dataset(),
#     boiling_cases()[1]['GOPR2878'].as_tf_dataset(),
#     boiling_cases()[2]['GOPR2908'].as_tf_dataset(),
#     boiling_cases()[3]['GOPR2948'].as_tf_dataset(),
# )
# reference_datasets_condensation = (
#     condensation_datasets_dict['stainless steel:polished'][
#         'stainless steel:polished:test 4:00003'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:old']['parametric:old:test 3:00006'].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_inf 40C'][
#         'parametric:T_inf 40C:test 2:00005'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_inf 60C'][
#         'parametric:T_inf 60C:test 1:00001'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_s 5C'][
#         'parametric:T_s 5C:test 3:00008'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:T_s 20C'][
#         'parametric:T_s 20C:test 3:00008'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:rh 70%'][
#         'parametric:rh 70%:test 3:00001'
#     ].as_tf_dataset(),
#     condensation_datasets_dict['parametric:rh 90%'][
#         'parametric:rh 90%:test 3:00009'
#     ].as_tf_dataset(),
# )

# """### FPS"""

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

# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Source: https://stackoverflow.com/questions/10917495/matplotlib-imshow-in-3d-plot

# # plt.clf()
# # fig = plt.figure(1)
# # ax = fig.gca(projection='3d')

# # params_no_reduce_lr = []
# # metrics_no_reduce_lr = []
# # params_reduce_lr = []
# # metrics_reduce_lr = []

# # for params, metrics in evaluations:
# #     if params['reduce_lr_on_plateau']:
# #         params_reduce_lr.append(params)
# #         metrics_reduce_lr.append(metrics)
# #     else:
# #         params_no_reduce_lr.append(params)
# #         metrics_no_reduce_lr.append(metrics)

# # X_label = 'batch_size'
# # Y_label = 'lr'
# # Z_label = 'MAE'

# # for reduce_lr in (False, True):
# #     params = {
# #         False: params_no_reduce_lr,
# #         True: params_reduce_lr
# #     }[reduce_lr]

# #     metrics = {
# #         False: metrics_no_reduce_lr,
# #         True: metrics_reduce_lr
# #     }[reduce_lr]

# #     X_list = list(set(funcy.pluck(X_label, params)))
# #     Y_list = list(set(funcy.pluck(Y_label, params)))

# #     X = np.zeros((len(X_list),))
# #     Y = np.zeros((len(Y_list),))
# #     Z = np.zeros((X.shape[0], Y.shape[0]))

# #     for (i, x), (j, y) in it.product(
# #             enumerate(X_list),
# #             enumerate(Y_list)
# #     ):
# #         X[i] = x
# #         Y[j] = y


# #         Z[i, j] =

# #     Z = np.array(
# #         funcy.pluck(
# #             Z_label,
# #             funcy.pluck(
# #                 'ds_val_gt10',
# #                 metrics
# #             )
# #         )
# #     )

# #     ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

# #     cset = ax.contourf(X, Y, Z, zdir='z', offset=min(Z),
# #             levels=np.linspace(min(Z),max(Z),100),cmap=plt.cm.jet)
# #     cset = ax.contourf(X, Y, Z, zdir='x', offset=min(X), cmap=plt.cm.jet)
# #     cset = ax.contourf(X, Y, Z, zdir='y', offset=max(Y), cmap=plt.cm.jet)

# #     ax.set_xlabel(X_label)
# #     ax.set_xlim(min(X), max(X))
# #     ax.set_ylabel(Y_label)
# #     ax.set_ylim(min(Y), max(Y))
# #     ax.set_zlabel(Z_label)
# #     ax.set_zlim(min(Z), max(Z))

# #     plt.show()

# """### Learning curve

# Evaluate models trained with different fractions of the dataset.
# For instance, 1%, 5%, 10%, 50% and 100% of the data.

# Tasks:

# - do some research: search for papers and books that explain this.

# ### Cross evaluation
# """

# metric_name = 'MAE'
# bar_metrics = dict(evaluations)
# bar_metrics = {
#     pack_: {set_name: set_metrics[metric_name] for set_name, set_metrics in sets.items()}
#     for pack_, sets in bar_metrics.items()
# }

# for k, v in bar_metrics.items():
#     print(k)
#     pprint(v)

# new_bar_metrics = {}
# for pack_, metrics_ in bar_metrics.items():
#     print('Pack:', pack_)
#     key = pack_.omit(['lr'])
#     print(key)
#     if key not in new_bar_metrics:
#         print('key not in new_bar_metrics')
#         new_bar_metrics[key] = {}
#     print(metrics_)
#     new_bar_metrics[key][pack_['lr']] = metrics_

# for pack_, metrics_dict in new_bar_metrics.items():
#     new_bar_metrics[pack_] = max(metrics_dict.values(), key=operator.itemgetter('ds_eval'))

# bar_metrics = {}
# for pack_, metrics_ in new_bar_metrics.items():
#     key = pack_.omit(['direct'])
#     print(key)
#     if key not in bar_metrics:
#         print('key not in bar_metrics')
#         bar_metrics[key] = {}
#     print(metrics_)
#     bar_metrics[key][pack_['direct']] = metrics_

# pprint(bar_metrics)

# # %matplotlib inline

# sns.set_context('notebook')
# sns.set_theme(style='ticks')

# metrics_data = pd.DataFrame.from_records(
#     [
#         {
#             'train': '+'.join(case_name[-1] for case_name in pack_['train_cases']),
#             'eval': '+'.join(case_name[-1] for case_name in pack_['eval_cases']),
#             'visualization': 'direct' if direct else 'indirect',
#             'metric': metrics_['ds_eval'],
#         }
#         for pack_, metrics_dict in bar_metrics.items()
#         for direct, metrics_ in metrics_dict.items()
#     ]
# )

# tips = sns.load_dataset('tips')

# # weird workaround, I don't know why metrics_data gives an error...
# new_metrics_data = tips.sample(len(metrics_data)).copy()
# new_metrics_data = new_metrics_data.rename(
#     columns={
#         'sex': 'train',
#         'total_bill': 'metric',
#         'smoker': 'eval',
#         'time': 'direct',
#     }
# )
# new_metrics_data['train'] = metrics_data['train']
# new_metrics_data['metric'] = metrics_data['metric']
# new_metrics_data['eval'] = metrics_data['eval']
# new_metrics_data['visualization'] = metrics_data['visualization']

# orient = 'v'
# x, y = ('train', 'metric')
# if orient == 'h':
#     x, y = y, x

# g = sns.catplot(
#     x=x,
#     y=y,
#     col='visualization',
#     data=new_metrics_data,
#     kind='bar',
#     hue='eval',
#     orient=orient,
# )
# label = 'MAE [W/cm²]'
# if orient == 'h':
#     g.set_xlabels(label)
# else:
#     g.set_ylabels(label)

# for ax in g.axes_dict.values():
#     for p, txt in zip(
#         ax.patches,
#         mit.flatten(
#             [
#                 itertools.repeat(txt, len(ax.patches) // len(g.legend.get_texts()))
#                 for txt in g.legend.get_texts()
#             ]
#         ),
#     ):
#         ax.text(
#             # p.get_x() - 0.01,
#             p.get_x(),
#             p.get_height() * 1.02,
#             txt.get_text(),
#             color='black',
#             rotation='horizontal',
#             size='large',
#         )


# def main(evaluations, metric_name: str):
#     bar_metrics = dict(evaluations)
#     bar_metrics = {
#         pack_: {set_name: set_metrics[metric_name] for set_name, set_metrics in sets.items()}
#         for pack_, sets in bar_metrics.items()
#     }

#     for k, v in bar_metrics.items():
#         print(k)
#         pprint(v)

#     new_bar_metrics = {}
#     for pack_, metrics_ in bar_metrics.items():
#         print('Pack:', pack_)
#         key = pack_.omit(['lr'])
#         print(key)
#         if key not in new_bar_metrics:
#             print('key not in new_bar_metrics')
#             new_bar_metrics[key] = {}
#         print(metrics_)
#         new_bar_metrics[key][pack_['lr']] = metrics_

#     for pack_, metrics_dict in new_bar_metrics.items():
#         new_bar_metrics[pack_] = max(metrics_dict.values(), key=operator.itemgetter('ds_eval'))

#     bar_metrics = {}
#     for pack_, metrics_ in new_bar_metrics.items():
#         key = pack_.omit(['direct'])
#         print(key)
#         if key not in bar_metrics:
#             print('key not in bar_metrics')
#             bar_metrics[key] = {}
#         print(metrics_)
#         bar_metrics[key][pack_['direct']] = metrics_

#     pprint(bar_metrics)

#     # %matplotlib inline

#     sns.set_context('notebook')
#     sns.set_theme(style='ticks')

#     metrics_data = pd.DataFrame.from_records(
#         [
#             {
#                 'train': '+'.join(case_name[-1] for case_name in pack_['train_cases']),
#                 'eval': '+'.join(case_name[-1] for case_name in pack_['eval_cases']),
#                 'visualization': 'direct' if direct else 'indirect',
#                 'metric': metrics_['ds_eval'],
#             }
#             for pack_, metrics_dict in bar_metrics.items()
#             for direct, metrics_ in metrics_dict.items()
#         ]
#     )

#     tips = sns.load_dataset('tips')

#     # weird workaround, I don't know why metrics_data gives an error...
#     new_metrics_data = tips.sample(len(metrics_data)).copy()
#     new_metrics_data = new_metrics_data.rename(
#         columns={
#             'sex': 'train',
#             'total_bill': 'metric',
#             'smoker': 'eval',
#             'time': 'direct',
#         }
#     )
#     new_metrics_data['train'] = metrics_data['train']
#     new_metrics_data['metric'] = metrics_data['metric']
#     new_metrics_data['eval'] = metrics_data['eval']
#     new_metrics_data['visualization'] = metrics_data['visualization']

#     orient = 'v'
#     x, y = ('train', 'metric')
#     if orient == 'h':
#         x, y = y, x
#     g = sns.catplot(
#         x=x,
#         y=y,
#         col='visualization',
#         data=new_metrics_data,
#         kind='bar',
#         hue='eval',
#         orient=orient,
#     )
#     label = 'MAE [W/cm²]'
#     if orient == 'h':
#         g.set_xlabels(label)
#     else:
#         g.set_ylabels(label)

#     for ax in g.axes_dict.values():
#         for p, txt in zip(
#             ax.patches,
#             mit.flatten(
#                 [
#                     itertools.repeat(txt, len(ax.patches) // len(g.legend.get_texts()))
#                     for txt in g.legend.get_texts()
#                 ]
#             ),
#         ):
#             ax.text(
#                 # p.get_x() - 0.01,
#                 p.get_x(),
#                 p.get_height() * 1.02,
#                 txt.get_text(),
#                 color='black',
#                 rotation='horizontal',
#                 size='large',
#             )


# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline

# sns.set_context('notebook')
# sns.set_theme(style='ticks')


# main(evaluations, 'MAE')

# """2+3 uses too much data!!!
# perhaps 2+3 could be defined as 50% of case 2 + 50% of case 3?
# """

# # # Fix the x-axes.
# # ax.set_xticks(x + bar_width)
# # ax.set_xticks(x)
# # ax.set_xticklabels(
# #     [
# #         '\n'.join([
# #             'train: ' + '+'.join(case_name[-1] for case_name in pack_['train_cases']),
# #             'eval: ' + '+'.join(case_name[-1] for case_name in pack_['eval_cases'])
# #         ])
# #         for pack_ in bar_metrics
# #     ],
# #     fontdict=dict(fontsize=10)
# # )

# # # Add legend.
# # ax.legend()

# # # Axis styling.
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_color('#DDDDDD')
# # ax.tick_params(bottom=False, left=False)
# # ax.set_axisbelow(True)
# # ax.yaxis.grid(True, color='#EEEEEE')
# # ax.xaxis.grid(False)

# # # Add axis and chart labels.
# # ax.set_xlabel('Job', labelpad=15)
# # ax.set_ylabel('MAE [W/cm²]', labelpad=15)
# # ax.set_title(f'Cross-evaluation metric', pad=15)

# # fig.tight_layout()
# # fig.show()

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
