import contextlib
import gc
import typing
from typing import Any, Optional, TypedDict

import autokeras as ak
import keras_tuner as kt
import tensorflow as tf
from loguru import logger
from tensorflow.keras import backend as K

from boiling_learning.utils.pathutils import resolve


class PopulateSpaceReturn(TypedDict):
    status: kt.engine.trial.TrialStatus
    values: Optional[dict[str, Any]]


class EarlyStoppingGreedyOracle(ak.tuners.greedy.GreedyOracle):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stop_search = False

    def populate_space(self, trial_id: str) -> PopulateSpaceReturn:
        if self.stop_search:
            return {
                'status': kt.engine.trial.TrialStatus.STOPPED,
                'values': None,
            }
        return typing.cast(PopulateSpaceReturn, super().populate_space(trial_id))

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(stop_search=self.stop_search)
        return typing.cast(dict[str, Any], state)

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self.stop_search = state['stop_search']


class EarlyStoppingHyperbandOracle(kt.oracles.HyperbandOracle):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stop_search = False

    def populate_space(self, trial_id: str) -> PopulateSpaceReturn:
        if self.stop_search:
            return {
                'status': kt.engine.trial.TrialStatus.STOPPED,
                'values': None,
            }
        return typing.cast(PopulateSpaceReturn, super().populate_space(trial_id))

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(stop_search=self.stop_search)
        return typing.cast(dict[str, Any], state)

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self.stop_search = state['stop_search']


class _SaveBestModelAtTrainingEndTuner(ak.engine.tuner.AutoTuner):
    """Only save models at the end of the training.

    If early stopping was defined as a callback, replace ``KerasTuner``'s ``SaveBestEpoch`` with
    a similar, custom implementation that only saves models at the end of the training.
    """

    def _build_and_fit_model(
        self, trial: kt.engine.trial.Trial, *fit_args: Any, **fit_kwargs: Any
    ) -> dict[str, list[Any]]:
        if 'callbacks' in fit_kwargs:
            callbacks = fit_kwargs['callbacks']

            if any(
                isinstance(callback, tf.keras.callbacks.EarlyStopping) for callback in callbacks
            ):
                index = next(
                    (
                        index
                        for index, callback in enumerate(callbacks)
                        if isinstance(callback, kt.engine.tuner_utils.SaveBestEpoch)
                    ),
                    None,
                )
                if index is not None:
                    callbacks.pop(index)
                    callbacks.insert(
                        index,
                        SaveBestEpoch(filepath=self._get_checkpoint_fname(trial.trial_id)),
                    )
                    fit_kwargs['callbacks'] = callbacks

        return typing.cast(
            dict[str, list[Any]],
            super()._build_and_fit_model(trial, *fit_args, **fit_kwargs),
        )

    def _get_checkpoint_fname(self, trial_id: str) -> str:
        return str(
            resolve(super()._get_checkpoint_fname(trial_id), parents=True).with_suffix('.h5')
        )


class _FixedMaxModelSizeTuner(_SaveBestModelAtTrainingEndTuner):
    def on_trial_end(self, trial: kt.engine.trial.Trial) -> None:
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if trial.get_state().get('status') != kt.engine.trial.TrialStatus.INVALID:
            self.oracle.end_trial(trial.trial_id, kt.engine.trial.TrialStatus.COMPLETED)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def _build_and_fit_model(
        self, trial: kt.engine.trial.Trial, *fit_args: Any, **fit_kwargs: Any
    ) -> dict[str, list[Any]]:
        with contextlib.suppress(tf.errors.ResourceExhaustedError, tf.errors.InternalError):
            with self.distribution_strategy.scope():
                model = self.hypermodel.build(trial.hyperparameters)

            model_size = self.maybe_compute_model_size(model)
            max_model_size_message = (
                'no maximum'
                if self.max_model_size is None
                else f'{model_size/self.max_model_size:.0%} {self.max_model_size}'
            )

            if self.max_model_size is None or model_size <= self.max_model_size:
                logger.info(f'Building model with size: {model_size} ({max_model_size_message})')

                # may be required to avoid errors:
                # fit_kwargs["callbacks"].extend(<your callbacks>)

                return super()._build_and_fit_model(trial, *fit_args, **fit_kwargs)

            logger.info(f'Skipping model with size: {model_size} ({max_model_size_message})')

        self.oracle.end_trial(trial.trial_id, kt.engine.trial.TrialStatus.INVALID)

        dummy_history_obj = tf.keras.callbacks.History()
        dummy_history_obj.on_train_begin()
        dummy_history_obj.history.setdefault('val_loss', []).append(_HUGE_NUMBER)
        return typing.cast(dict[str, list[Any]], dummy_history_obj)

    def _try_build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        # clean-up TF graph from previously stored (defunct) graph
        K.clear_session()
        gc.collect()

        # Build a model - failed attempts are handled elsewhere
        model = self._build_hypermodel(hp)

        # Stop if `build()` does not return a valid model.
        if not isinstance(model, tf.keras.models.Model):
            raise RuntimeError(
                'Model-building function did not return a valid Keras Model instance, '
                f'found {model}'
            )

        return model

    def maybe_compute_model_size(self, model: tf.keras.models.Model) -> int:
        """Compute the size of a given model, if it has been built."""
        if model.built:
            return sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
        return 0


class EarlyStoppingGreedy(_FixedMaxModelSizeTuner):
    def __init__(
        self,
        *,
        goal: Any = None,
        objective: str = 'val_loss',
        max_trials: int = 10,
        initial_hps: Optional[list[dict[str, Any]]] = None,
        seed: Optional[int] = None,
        hyperparameters: Optional[kt.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs: Any,
    ) -> None:
        self.goal = goal
        oracle = EarlyStoppingGreedyOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle=oracle, **kwargs)

    def on_epoch_end(
        self,
        trial: kt.engine.trial.Trial,
        model: tf.keras.models.Model,
        epoch: str,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().on_epoch_end(trial, model, epoch, logs)

        if self.goal is not None:
            objective = self.oracle.objective
            loss = objective.get_value(logs)

            if objective.better_than(loss, self.goal):
                logger.info(f'Got {loss}, and the desired objective is {self.goal}. Stopping now.')
                model.stop_training = True
                self.oracle.stop_search = True


class EarlyStoppingHyperband(_FixedMaxModelSizeTuner):
    def __init__(
        self,
        *,
        goal: Any = None,
        objective: str = 'val_loss',
        max_epochs: int = 100,
        max_trials: int = 1000,
        factor: int = 3,
        seed: Optional[int] = None,
        hyperparameters: Optional[kt.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs: Any,
    ) -> None:
        self.goal = goal
        oracle = EarlyStoppingHyperbandOracle(
            objective=objective,
            max_epochs=max_epochs,
            factor=factor,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        oracle.max_trials = max_trials
        super().__init__(oracle=oracle, **kwargs)

    def on_epoch_end(
        self,
        trial: kt.engine.trial.Trial,
        model: tf.keras.models.Model,
        epoch: str,
        logs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().on_epoch_end(trial, model, epoch, logs)

        if self.goal is not None:
            objective = self.oracle.objective
            loss = objective.get_value(logs)

            if objective.better_than(loss, self.goal):
                logger.info(f'Got {loss}, and the desired objective is {self.goal}. Stopping now.')
                model.stop_training = True
                self.oracle.stop_search = True


_HUGE_NUMBER = 100000.0


class SaveBestEpoch(tf.keras.callbacks.Callback):
    """A Keras callback to save the model weights at the end of the training."""

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        self.model.save_weights(self.filepath)
