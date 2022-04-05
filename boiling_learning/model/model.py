import enum
from operator import itemgetter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, TypeVar

import parse
import tensorflow as tf

from boiling_learning.io.io import LoaderFunction
from boiling_learning.io.storage import Metadata, deserialize, serialize
from boiling_learning.utils import PathLike, resolve

T = TypeVar('T')
_ModelType = TypeVar('_ModelType')


def restore(
    path_pattern: PathLike,
    load_method: LoaderFunction[T],
    epoch_str: str = 'epoch',
) -> Tuple[int, Optional[T]]:
    path_pattern = resolve(path_pattern)

    glob_pattern = path_pattern.name.replace(f'{{{epoch_str}}}', '*')
    paths = path_pattern.parent.glob(glob_pattern)

    parser = parse.compile(path_pattern.name).parse
    parsed = (parser(path_item.name) for path_item in paths)

    epochs = (int(p[epoch_str]) for p in parsed if p is not None and epoch_str in p)
    last_epoch = max(epochs, default=-1)

    if last_epoch == -1:
        return -1, None

    path_str = str(path_pattern).format(epoch=last_epoch)
    model = load_method(path_str)
    return last_epoch, model


class ProblemType(enum.Enum):
    CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()


def model_checkpoints(pattern: PathLike, *, epoch_key: str = 'epoch') -> Dict[int, Path]:
    pattern = resolve(pattern)
    filename_pattern = pattern.name
    glob_pattern = filename_pattern.replace(f'{{{epoch_key}}}', '*')
    parser = parse.compile(filename_pattern).parse

    paths = pattern.parent.glob(glob_pattern)

    path_dict = {parser(path.name): path for path in paths}

    return {
        int(parsed_obj[epoch_key]): path
        for parsed_obj, path in path_dict.items()
        if parsed_obj is not None and epoch_key in parsed_obj
    }


def models_from_checkpoints(
    pattern: PathLike,
    load_method: LoaderFunction[tf.keras.models.Model] = tf.keras.models.load_model,
    *,
    epoch_key: str = 'epoch',
) -> Dict[int, tf.keras.models.Model]:
    path_dict = model_checkpoints(pattern, epoch_key=epoch_key)

    return {epoch: load_method(str(path)) for epoch, path in path_dict.items()}


def last_model_checkpoint(
    pattern: PathLike,
    load_method: LoaderFunction[tf.keras.models.Model] = tf.keras.models.load_model,
    *,
    epoch_key: str = 'epoch',
) -> Optional[Tuple[int, tf.keras.models.Model]]:
    path_dict = model_checkpoints(pattern, epoch_key=epoch_key)

    last_pair = max(path_dict.items(), default=None, key=itemgetter(0))

    if last_pair is None:
        return None

    epoch, path = last_pair
    return epoch, load_method(path)


def history_from_checkpoints(
    ds_val: tf.data.Dataset,
    pattern: PathLike,
    load_method: LoaderFunction[tf.keras.models.Model] = tf.keras.models.load_model,
    *,
    epoch_key: str = 'epoch',
) -> Dict[int, Dict[str, float]]:
    model_dict = models_from_checkpoints(pattern, load_method, epoch_key=epoch_key)
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


class Model(tf.keras.models.Model):
    pass


@serialize.instance(Model)
def _serialize_model(instance: Model, path: Path) -> None:
    tf.keras.models.save_model(instance, resolve(path))


@deserialize.dispatch(Model)
def _deserialize_model(path: Path, _metadata: Metadata) -> Model:
    return tf.keras.models.load_model(path)
