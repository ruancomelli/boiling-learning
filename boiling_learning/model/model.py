import enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar

import funcy
import parse
import tensorflow as tf

from boiling_learning.io.io import LoaderFunction
from boiling_learning.preprocessing.transformers import Creator
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike, append, resolve

T = TypeVar('T')
_ModelType = TypeVar('_ModelType')


def restore(
    restore: bool = False,
    path: Optional[PathLike] = None,
    load_method: Optional[LoaderFunction[T]] = None,
    epoch_str: str = 'epoch',
) -> Tuple[int, Optional[T]]:
    last_epoch = -1
    model = None
    if restore:
        path = resolve(path)
        glob_pattern = path.name.replace(f'{{{epoch_str}}}', '*')
        parser = parse.compile(path.name).parse

        paths = path.parent.glob(glob_pattern)
        parsed = (parser(path_item.name) for path_item in paths)
        parsed = filter(lambda p: p is not None and epoch_str in p, parsed)
        epochs = append((int(p[epoch_str]) for p in parsed), last_epoch)
        last_epoch = max(epochs)

        if last_epoch != -1:
            path_str = str(path).format(epoch=last_epoch)
            model = load_method(path_str)

    return last_epoch, model


class ProblemType(enum.Enum):
    CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()


def default_compiler(model, **params):
    model.compile(**params)
    return model


def default_fitter(model, **params):
    return model.fit(**params)


def make_creator_method(
    builder: Callable[..., _ModelType],
    compiler: Callable[[_ModelType], _ModelType] = default_compiler,
    fitter: Callable[[_ModelType], Any] = default_fitter,
) -> Callable[..., dict]:
    def creator_method(
        num_classes,
        problem,
        strategy,
        architecture_setup,
        compile_setup,
        fit_setup,
        fetch,
    ):
        with strategy.scope():
            model = builder(problem=problem, num_classes=num_classes, **architecture_setup)

            if compile_setup.get('do', False):
                model = compiler(model, **compile_setup['params'])

        history = None
        if fit_setup.get('do', False):
            history = fitter(model, **fit_setup['params'])

        available_data = {'model': model, 'history': history}

        return {k: available_data[k] for k in fetch}

    return creator_method


def make_creator(name: str, defaults: Pack = Pack()) -> Callable[[Callable], Callable]:
    return funcy.compose(
        Creator.make(name, pack=defaults, expand_pack_on_call=True),
        make_creator_method,
    )


def models_from_checkpoints(
    pattern: PathLike,
    epoch_key: str = 'epoch',
    load_method: LoaderFunction[tf.keras.models.Model] = tf.keras.models.load_model,
) -> Dict[int, tf.keras.models.Model]:
    pattern = resolve(pattern)
    filename_pattern = pattern.name
    glob_pattern = filename_pattern.replace(f'{{{epoch_key}}}', '*')
    parser = parse.compile(filename_pattern).parse

    paths = pattern.parent.glob(glob_pattern)

    path_dict = {parser(path.name): path for path in paths}
    path_dict = {
        int(parsed_obj[epoch_key]): path
        for parsed_obj, path in path_dict.items()
        if parsed_obj is not None and epoch_key in parsed_obj
    }

    return {epoch: load_method(str(path)) for epoch, path in path_dict.items()}


def history_from_checkpoints(
    ds_val: tf.data.Dataset,
    pattern: PathLike,
    epoch_key: str = 'epoch',
    load_method: LoaderFunction[tf.keras.models.Model] = tf.keras.models.load_model,
) -> Dict[int, Dict[str, float]]:
    model_dict = models_from_checkpoints(pattern, epoch_key, load_method)
    return {epoch: model.evaluate(ds_val, return_dict=True) for epoch, model in model_dict.items()}


def eval_with(
    model: tf.keras.models.Model,
    ds_val: tf.data.Dataset,
    metrics: Iterable[tf.keras.metrics.Metric],
    reset_state: bool = True,
) -> Dict[str, float]:
    metrics = tuple(metrics)

    if reset_state:
        for metric in metrics:
            metric.reset_states()

    for metric in metrics:
        for x, y_true in ds_val:
            y_pred = model.predict(x, use_multiprocessing=True, workers=-1)
            metric.update_state(y_true, y_pred)

    return {metric.name: metric.result().numpy() for metric in metrics}
