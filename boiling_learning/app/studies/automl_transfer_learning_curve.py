from fractions import Fraction
from typing import Literal

import tensorflow as tf
import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import best_baseline_boiling1d_model
from boiling_learning.app.cancellation import CancelledError
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_baseline_model_size,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler

app = typer.Typer()
console = Console()

FRACTIONS = (0, Fraction(1, 100), Fraction(1, 10), 1)


@app.command()
# @logger.catch
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    raise CancelledError(
        "This study is cancelled for not providing useful information."
    )

    model_evaluator = cached_model_evaluator("boiling1d")

    tables: list[Table] = []
    for direct in False, True:
        direct_label = "direct" if direct else "indirect"

        datasets = boiling_datasets(direct_visualization=direct)[1]

        table = Table(
            "Subsample",
            "Training\nloss",
            "Validation\nloss",
            "Test\nloss",
            "Epochs\ntrained",
            title=f"Learning curve - {direct_label}",
        )

        for fraction in FRACTIONS:
            best_model = best_baseline_boiling1d_model(
                direct_visualization=direct,
                strategy=strategy,
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                normalize_images=True,
                max_model_size=get_baseline_model_size(
                    direct_visualization=direct,
                    strategy=strategy,
                ),
            )

            if fraction:
                subsampled = (
                    datasets | dataset_sampler(count=fraction, subset="train")
                    if fraction != 1
                    else datasets
                )

                compiled_pretrained_model = best_model | compile_model(
                    **get_baseline_compile_params(strategy=strategy),
                )
                fine_tuned_model_return = fit_boiling_model(
                    compiled_pretrained_model,
                    subsampled,
                    get_baseline_fit_params(batch_size=32),
                    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                    strategy=strategy,
                )
                fine_tuned_model = fine_tuned_model_return.architecture
                trained_epochs = fine_tuned_model_return.trained_epochs
            else:
                fine_tuned_model = best_model
                trained_epochs = 0

            compiled_model = fine_tuned_model | compile_model(
                **get_baseline_compile_params(strategy=strategy)
            )

            evaluation = model_evaluator(
                compiled_model,
                datasets,
                measure_uncertainty=False,
            )

            table.add_row(
                f"{fraction} ({float(fraction):.0%})",
                f"{evaluation.training_metrics['MSE']:.2f}",
                f"{evaluation.validation_metrics['MSE']:.2f}",
                f"{evaluation.test_metrics['MSE']:.2f}",
                str(trained_epochs),
            )

        tables.append(table)

        table = Table(
            "Learning rate",
            "Freezing",
            "Training\nloss",
            "Validation\nloss",
            "Test\nloss",
            "Epochs\ntrained",
            title=f"Learning curve - 1% - {direct_label}",
        )

        subsampled = datasets | dataset_sampler(count=Fraction(1, 100), subset="train")
        for learning_rate in 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001:
            for freezing in "none", "pre", "body":
                best_model = best_baseline_boiling1d_model(
                    direct_visualization=direct,
                    strategy=strategy,
                    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                    normalize_images=True,
                    max_model_size=get_baseline_model_size(
                        direct_visualization=direct,
                        strategy=strategy,
                    ),
                )

                frozen_model = _freeze(
                    best_model,
                    strategy=strategy,
                    freezing=freezing,
                )

                compiled_frozen_model = frozen_model | compile_model(
                    **get_baseline_compile_params(
                        strategy=strategy,
                        learning_rate=learning_rate,
                    ),
                )

                fit_model = fit_boiling_model(
                    compiled_frozen_model,
                    subsampled,
                    get_baseline_fit_params(batch_size=32),
                    target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                    strategy=strategy,
                )
                trained_epochs = fit_model.trained_epochs

                compiled_model = fit_model.architecture | compile_model(
                    **get_baseline_compile_params(strategy=strategy)
                )

                evaluation = model_evaluator(
                    compiled_model,
                    datasets,
                    measure_uncertainty=False,
                )

                table.add_row(
                    str(learning_rate),
                    freezing,
                    f"{evaluation.training_metrics['MSE']:.2f}",
                    f"{evaluation.validation_metrics['MSE']:.2f}",
                    f"{evaluation.test_metrics['MSE']:.2f}",
                    str(trained_epochs),
                )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _freeze(
    model: LazyDescribed[ModelArchitecture],
    strategy: LazyDescribed[tf.distribute.Strategy],
    freezing: Literal["none", "pre", "body"],
) -> LazyDescribed[ModelArchitecture]:
    architecture = model().clone(strategy=strategy)

    if freezing == "body":
        architecture.model.trainable = False
        architecture.model.layers[-1].trainable = True
    elif freezing == "pre":
        architecture.model.trainable = False
        architecture.model.layers[-2].trainable = True
        architecture.model.layers[-1].trainable = True

    return LazyDescribed.from_describable(architecture)
