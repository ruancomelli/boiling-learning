import contextlib
import gc
import typing
from typing import Any, Dict, List, Optional

import autokeras as ak
import keras_tuner as kt
import tensorflow as tf
from loguru import logger
from tensorflow.keras import backend as K
from typing_extensions import TypedDict


class PopulateSpaceReturn(TypedDict):
    status: kt.engine.trial.TrialStatus
    values: Optional[Dict[str, Any]]


class EarlyStoppingGreedyOracle(ak.tuners.greedy.GreedyOracle):
    def __init__(self, *, goal: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stop_search = False
        self.goal = goal

    def populate_space(self, trial_id: str) -> PopulateSpaceReturn:
        if self.stop_search:
            return {
                'status': kt.engine.trial.TrialStatus.STOPPED,
                'values': None,
            }
        return typing.cast(PopulateSpaceReturn, super().populate_space(trial_id))

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update(stop_search=self.stop_search, goal=self.goal)
        return typing.cast(Dict[str, Any], state)

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state)
        self.stop_search = state['stop_search']
        self.goal = state['goal']


class _NoAutomaticSaveBestModel(ak.engine.tuner.AutoTuner):
    def _build_and_fit_model(
        self, trial: kt.engine.trial.Trial, *fit_args: Any, **fit_kwargs: Any
    ) -> Dict[str, List[Any]]:
        if 'callbacks' in fit_kwargs:
            fit_kwargs['callbacks'] = [
                callback
                for callback in fit_kwargs['callbacks']
                if not isinstance(callback, kt.engine.tuner_utils.SaveBestEpoch)
            ]

        return typing.cast(
            Dict[str, List[Any]],
            super()._build_and_fit_model(trial, *fit_args, **fit_kwargs),
        )


class _FixedMaxModelSizeGreedy(_NoAutomaticSaveBestModel):
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
    ) -> Dict[str, List[Any]]:
        with contextlib.suppress(tf.errors.ResourceExhaustedError, tf.errors.InternalError):
            with self.distribution_strategy.scope():
                model = self.hypermodel.build(trial.hyperparameters)

            model_size = self.maybe_compute_model_size(model)

            if self.max_model_size is None or model_size <= self.max_model_size:
                logger.info(f'Building model with size: {model_size}')

                # TODO: may be required to avoid errors:
                # fit_kwargs["callbacks"].extend(<your callbacks>)

                return super()._build_and_fit_model(trial, *fit_args, **fit_kwargs)

            logger.info(f'Skipping model with size: {model_size}')

        self.oracle.end_trial(trial.trial_id, kt.engine.trial.TrialStatus.INVALID)

        dummy_history_obj = tf.keras.callbacks.History()
        dummy_history_obj.on_train_begin()
        dummy_history_obj.history.setdefault('val_loss', []).append(_HUGE_NUMBER)
        return typing.cast(Dict[str, List[Any]], dummy_history_obj)

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


class EarlyStoppingGreedy(_FixedMaxModelSizeGreedy):
    def __init__(
        self,
        *,
        goal: Any,
        objective: str = 'val_loss',
        max_trials: int = 10,
        initial_hps: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        hyperparameters: Optional[kt.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs: Any,
    ) -> None:
        self.seed = seed
        oracle = EarlyStoppingGreedyOracle(
            objective=objective,
            goal=goal,
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
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().on_epoch_end(trial, model, epoch, logs)

        objective = self.oracle.objective
        loss = objective.get_value(logs)

        if objective.better_than(loss, self.oracle.goal):
            logger.info(
                f'Got {loss}, and the desired objective is {self.oracle.goal}. Stopping now.'
            )
            model.stop_training = True
            self.oracle.stop_search = True


_HUGE_NUMBER = 100000.0
