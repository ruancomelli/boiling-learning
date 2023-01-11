from fractions import Fraction
from pathlib import Path

import tensorflow as tf
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import evaluate_boiling_model_with_dataset
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler

app = typer.Typer()
console = Console()

FRACTIONS = (0, Fraction(1, 100), Fraction(1, 10), 1)


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
) -> None:
    logger.info('Analyzing learning curve')

    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    datasets = boiling_datasets(direct_visualization=direct)[1]

    validation_losses: list[float] = []

    table = Table(
        'Subsample',
        'Validation\nloss',
        'Test\nloss',
        'Epochs\ntrained',
        title='Learning curve',
    )

    for fraction in FRACTIONS:
        subsampled = (
            datasets | dataset_sampler(count=fraction, subset='train')
            if fraction != 1
            else datasets
        )

        pretrained_model = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=True,
            strategy=strategy,
        )

        compiled_model = pretrained_model.architecture | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )
        if fraction:
            fit_model = fit_boiling_model(
                compiled_model,
                subsampled,
                get_baseline_fit_params(),
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                strategy=strategy,
            )

            trained_epochs = fit_model.trained_epochs
            validation_metrics = fit_model.validation_metrics
            test_metrics = fit_model.test_metrics
        else:
            _, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
                compiled_model,
                datasets,
            )

            trained_epochs = 0
            validation_metrics = {
                metric_name: metric.value for metric_name, metric in validation_metrics.items()
            }
            test_metrics = {
                metric_name: metric.value for metric_name, metric in test_metrics.items()
            }

        table.add_row(
            f'{fraction} ({float(fraction):.0%})',
            f'{validation_metrics["MSE"]:.2f}',
            f'{test_metrics["MSE"]:.2f}',
            str(trained_epochs),
        )

        validation_losses.append(validation_metrics['MSE'])

    console.print(table)

    # TODO: fix this
    # f, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.scatter(list(map(float, FRACTIONS)), validation_losses)
    # ax.set_xticks(list(map(float, FRACTIONS)))
    # ax.set_xticklabels(FRACTIONS)
    # ax.set_xlabel('Dataset subsample')
    # ax.set_ylabel('Validation loss')
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # figure_path = resolve(
    #     _learning_curve_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png",
    #     parents=True,
    # )
    # f.savefig(str(figure_path))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _learning_curve_study_path() -> Path:
    return studies_path() / 'learning-curve'


def _freeze(
    model: LazyDescribed[ModelArchitecture],
    strategy: LazyDescribed[tf.distribute.Strategy],
    body: bool,
    pre: bool,
) -> LazyDescribed[ModelArchitecture]:
    architecture = model().clone(strategy=strategy)

    if body:
        architecture.model.trainable = False

    if pre:
        architecture.model.layers[-2].trainable = False

    architecture.model.layers[-1].trainable = True
    return LazyDescribed.from_describable(architecture)
