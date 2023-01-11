from operator import itemgetter

import tensorflow as tf
from loguru import logger

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.app.datasets.raw.condensation import condensation_datasets
from boiling_learning.datasets.sliceable import map_targets, targets
from boiling_learning.distribute import strategy_scope
from boiling_learning.image_datasets import Targets
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.definitions import hoboldnet2
from boiling_learning.model.training import compile_model
from boiling_learning.preprocessing.experiment_video_dataset import ExperimentVideoDataset

# TODO: check
# <https://stackoverflow.com/a/58970598/5811400>
# and
# <https://github.com/googlecolab/colabtools/issues/864#issuecomment-556437040>
# TODO: na condensação, fazer crop determinístico!!!
# TODO: depois, se tiver tempo, fazer RandomCrop pra comparar
# TODO: ver o quanto influencia crop determinístico versus randomico
# TODO: para ambos os tipos de corte, rodar auto ML
# TODO: para um mesmo fluxo, mostrar imagens para os quatro datasets
# TODO: fazer vídeos tipo o do Hobold com a ebulição e um gráfico (barrinha de erro), o
# fluxo nominal e o valor predito
# TODO: esse vídeo pode ser para as quatro superfícies ao mesmo tempo
# TODO: generate a test case in which temperature is estimated from the boiling curve and that's
# what the models have to predict

# TODO: fazer erro em função do y: ver se para maiores ys o erro vai subindo ou diminuindo
# quem sabe fazer 3 ou mais modelos, um especializado para cada região de y; e quem sabe
# usar um classificador pra escolher qual estimador não ajude muito
# focar na arquitetura da rede, que é mais importante do que hiperparâmetros
# otimizar as convolucionais pode ser mais importante do que otimizar as fully-connected


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


"""#### First Condensation Case"""
strategy = configure(
    force_gpu_allow_growth=True,
    use_xla=True,
    # mixed_precision_global_policy='mixed_float16',
    require_gpu=True,
)

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
    compiled_model = architecture | compile_model(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name='top3'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name='top5'),
            # tfa.metrics.F1Score(N_CLASSES, name='F1'),
        ],
    )

assert False, 'STOP!'
