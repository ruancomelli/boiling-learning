import datetime
import gc
import shutil
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from loguru import logger
from tensorflow.data import Dataset
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import BackupAndRestore as _BackupAndRestore
from tensorflow.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging
from typing_extensions import Literal

from boiling_learning.io import json
from boiling_learning.utils import PathLike, resolve


# Source: <https://stackoverflow.com/q/47731935/5811400>
class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets: Dict[str, Dataset]) -> None:
        """
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super().__init__()

        self.validation_sets = validation_sets
        self.history = {}

    def on_train_begin(self, logs=None) -> None:
        self.history = {}

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set_name, validation_set in self.validation_sets.items():
            try:
                logger.info(f'Evaluating model on additional dataset {validation_set_name}')

                results = self.model.evaluate(validation_set)

                metric_names = ['loss'] + [m.name for m in self.model.metrics]

                full_names = [
                    f'{validation_set_name}_{metric_name}' for metric_name in metric_names
                ]
                full_results = [logs['loss']] + results
                for full_name, result in zip(full_names, full_results):
                    self.history.setdefault(full_name, []).append(result)

                values_str = ' - '.join(
                    f'{name}: {result}' for name, result in zip(metric_names, full_results)
                )
                logger.info(f'{validation_set_name}[{values_str}]')
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.info(e)


class TimePrinter(Callback):
    def __init__(
        self,
        streamer: Callable[[str], None] = print,
        fmt: str = '%Y-%m-%d %H:%M:%S',
        when: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        self.streamer = streamer
        self.fmt = fmt
        self._current_epoch = 0
        self.start: datetime.datetime

        if when is None:
            when = {
                'on_batch_begin',
                'on_batch_end',
                'on_epoch_begin',
                'on_epoch_end',
                'on_predict_batch_begin',
                'on_predict_batch_end',
                'on_predict_begin',
                'on_predict_end',
                'on_test_batch_begin',
                'on_test_batch_end',
                'on_test_begin',
                'on_test_end',
                'on_train_batch_begin',
                'on_train_batch_end',
                'on_train_begin',
                'on_train_end',
            }
        self.when = frozenset(when)

    def _str_now(self) -> str:
        return datetime.datetime.now().strftime(self.fmt)

    def on_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        if 'on_batch_begin' in self.when:
            self.streamer(f'--- beginning batch at {self._str_now()}')

    def on_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if 'on_batch_end' in self.when:
            self.streamer(f' | ending batch at {self._str_now()}')

    def on_epoch_begin(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        self._current_epoch = epoch
        if 'on_epoch_begin' in self.when:
            self.streamer(f'-- beginning epoch {epoch + 1} at {self._str_now()}')

    def on_epoch_end(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        if 'on_epoch_end' in self.when:
            self.streamer(f'-- ending epoch {epoch + 1} at {self._str_now()}')

    def on_predict_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        if 'on_predict_batch_begin' in self.when:
            self.streamer(f'--- beginning predict_batch at {self._str_now()}')

    def on_predict_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if 'on_predict_batch_end' in self.when:
            self.streamer(f' | ending predict_batch at {self._str_now()}')

    def on_predict_begin(self, *args: Any, **kwargs: Any) -> None:
        if 'on_predict_begin' in self.when:
            self.streamer(f'- beginning predict at {self._str_now()}')

    def on_predict_end(self, *args: Any, **kwargs: Any) -> None:
        if 'on_predict_end' in self.when:
            self.streamer(f'- ending predict at {self._str_now()}')

    def on_test_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        if 'on_test_batch_begin' in self.when:
            self.streamer(f'--- beginning test_batch at {self._str_now()}')

    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if 'on_test_batch_end' in self.when:
            self.streamer(f' | ending test_batch at {self._str_now()}')

    def on_test_begin(self, *args: Any, **kwargs: Any) -> None:
        if 'on_test_begin' in self.when:
            self.streamer(f'- beginning test at {self._str_now()}')

    def on_test_end(self, *args: Any, **kwargs: Any) -> None:
        if 'on_test_end' in self.when:
            self.streamer(f'- ending test at {self._str_now()}')

    def on_train_batch_begin(self, batch: int, *args: Any, **kwargs: Any) -> None:
        if 'on_train_batch_begin' in self.when:
            self.streamer(
                f'--- epoch {self._current_epoch + 1}: '
                f'beginning train_batch {batch} at {self._str_now()}',
            )

    def on_train_batch_end(self, batch: int, *args: Any, **kwargs: Any) -> None:
        if 'on_train_batch_end' in self.when:
            self.streamer(f' | ending train_batch {batch} at {self._str_now()}')

    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        self.start = datetime.datetime.now()
        if 'on_train_begin' in self.when:
            self.streamer(f'- beginning train at {self._str_now()}')

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        end = datetime.datetime.now()
        duration = end - self.start
        if 'on_train_end' in self.when:
            self.streamer(f'- ending train at {self._str_now()}')
        self.streamer(f'Training took {duration.total_seconds()} seconds')


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Example:
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
        will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
        min_delta_mode: one of `{'absolute', 'relative'}`.
        cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        mode: Literal['auto', 'min', 'max'] = 'auto',
        min_delta: float = 1e-4,
        min_delta_mode: str = 'absolute',
        cooldown: int = 0,
        min_lr: float = 0,
    ) -> None:
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.min_delta_mode = min_delta_mode
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0.0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self) -> None:
        """Resets wait counter and cooldown counter."""
        if self.mode not in {'auto', 'min', 'max'}:
            logging.warning(
                'Learning rate reduction mode %s is unknown, ' 'fallback to auto mode.',
                self.mode,
            )
            self.mode = 'auto'

        if self.min_delta_mode not in {'absolute', 'relative'}:
            logging.warning(
                'Minimum delta mode %s is unknown, ' 'fallback to absolute mode.',
                self.min_delta_mode,
            )
            self.min_delta_mode = 'absolute'

        if self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor):
            if self.min_delta_mode == 'relative':
                self.monitor_op = lambda current, best: np.less(
                    current, (1 - self.min_delta) * best
                )
            else:
                self.monitor_op = lambda current, best: np.less(current, best - self.min_delta)
            self.best = np.Inf
        else:
            if self.min_delta_mode == 'relative':
                self.monitor_op = lambda current, best: np.greater(
                    current, (1 + self.min_delta) * best
                )
            else:
                self.monitor_op = lambda current, best: np.greater(current, best + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                'Learning rate reduction is conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor,
                ', '.join(logs.keys()),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
            if self.wait >= self.patience:
                old_lr = K.get_value(self.model.optimizer.lr)
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                logger.info(
                    f'Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr}'
                )
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class RegisterEpoch(Callback):
    def __init__(self, path: PathLike) -> None:
        self._path = resolve(path, parents=True)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self._path.write_text(str(epoch), encoding='utf8')

    def last_epoch(self) -> int:
        return int(self._path.read_text(encoding='utf8'))

    def __describe__(self) -> str:
        return str(self._path)


class SaveHistory(Callback):
    def __init__(self, path: PathLike, *, mode: Literal['a', 'w']) -> None:
        self._path = resolve(path, parents=True)

        self.history: List[Dict[str, Any]] = []

        if mode == 'a' and self._path.is_file():
            self.history.extend(json.load(self._path))

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        self._append_to_history(logs)
        json.dump(self.history, self._path)

    def __describe__(self) -> str:
        return str(self._path)

    def _append_to_history(self, logs: Dict[str, Any]) -> None:
        self.history.append({key: float(value) for key, value in logs.items()})


class BackupAndRestore(_BackupAndRestore):
    def __init__(self, backup_dir: PathLike, delete_on_end: bool = True) -> None:
        self.delete_on_end = delete_on_end
        backup_dir = resolve(backup_dir, dir=True)
        super().__init__(str(backup_dir))

    def on_train_end(self, logs=None):
        # Based on https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L1709-L1713

        if self.delete_on_end:
            shutil.rmtree(self.backup_dir)

        del self._training_state
        del self.model._training_state


class MemoryCleanUp(Callback):
    def __init__(self, clean_device: Optional[int] = None) -> None:
        self._clean_device = clean_device

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        # Housekeeping
        gc.collect()
        K.clear_session()

    # TODO: add numba as dependency and `from numba import cuda`
    # def on_train_end(self, logs=None) -> None:
    #     # Based on https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L1709-L1713

    #     if self._clean_device is not None:
    #         cuda.select_device(0)
    #         cuda.close()
