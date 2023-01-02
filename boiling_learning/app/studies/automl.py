import functools
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.condensation import get_pretrained_baseline_condensation_model
from boiling_learning.app.training.evaluation import evaluate_boiling_model_with_dataset
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler

app = typer.Typer()
console = Console()


@app.command()
def boiling1d(
    each: int = typer.Option(1),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    tables: list[Table] = []
    for direct in (False, True):
        table = Table(
            'Validation loss',
            'Test loss',
            title=(
                'Automatic machine learning - '
                + ('direct' if direct else 'indirect')
                + ' visualization'
            ),
        )

        baseline_fit_return = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=True,
            strategy=strategy,
        )

        baseline_loss = baseline_fit_return.validation_metrics['MSE']
        baseline_architecture_size = baseline_fit_return.architecture().count_parameters(
            trainable=True,
            non_trainable=False,
        )

        datasets = baseline_boiling_dataset(direct_visualization=direct)

        if each != 1:
            datasets = datasets | dataset_sampler(Fraction(1, each), subset='train')

        hypermodel = autofit_dataset(
            datasets,
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            normalize_images=True,
            max_model_size=baseline_architecture_size,
            goal=None,
            experiment='boiling1d',
            strategy=strategy,
        ).hypermodel

        compiled_model = LazyDescribed.from_describable(hypermodel.best_model()) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        _, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
            compiled_model,
            datasets,
        )

        table.add_row(
            str(validation_metrics['MSE']),
            str(test_metrics['MSE']),
        )
        tables.append(table)

        model_evaluator = _cached_model_evaluator('boiling1d')
        trainable_sizes = []
        total_sizes = []
        losses = []
        for model in hypermodel.iter_best_models():
            compiled_model = LazyDescribed.from_describable(model) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            trainable_size, total_size, validation_loss, _test_loss = model_evaluator(
                compiled_model,
                datasets,
            )

            trainable_sizes.append(trainable_size)
            total_sizes.append(total_size)
            losses.append(validation_loss)

        sns.set_style('whitegrid')

        save_path = (
            _automl_study_path()
            / f"boiling1d-{'direct' if direct else 'indirect'}-each-{each}-trainable.png"
        )
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(trainable_sizes, losses, s=20, color='k')
        ax.scatter(
            baseline_architecture_size,
            baseline_loss,
            facecolors='none',
            edgecolors='k',
            marker='$\\odot$',
            s=100,
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Model size')
        ax.set_ylabel('Validation loss')
        f.savefig(str(save_path))

        save_path = (
            _automl_study_path()
            / f"boiling1d-{'direct' if direct else 'indirect'}-each-{each}-total.png"
        )
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(total_sizes, losses, s=20, color='k')
        ax.scatter(
            baseline_architecture_size,
            baseline_loss,
            facecolors='none',
            edgecolors='k',
            marker='$\\odot$',
            s=100,
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Model size')
        ax.set_ylabel('Validation loss')
        f.savefig(str(save_path))

    console.print(Columns(tables))


@app.command()
def condensation(
    each: int = typer.Option(1),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    table = Table(
        'Validation loss',
        'Test loss',
        title='Automatic machine learning - condensation',
    )

    baseline_fit_return = get_pretrained_baseline_condensation_model(
        each=each,
        normalize_images=True,
        strategy=strategy,
    )

    baseline_loss = baseline_fit_return.validation_metrics['MSE']
    baseline_architecture_size = baseline_fit_return.architecture().count_parameters(
        trainable=True,
        non_trainable=False,
    )

    datasets = condensation_dataset(each=each)

    if each != 1:
        datasets = datasets | dataset_sampler(Fraction(1, each), subset='train')

    hypermodel = autofit_dataset(
        datasets,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        normalize_images=True,
        max_model_size=baseline_architecture_size,
        goal=None,
        experiment='condensation',
        strategy=strategy,
    ).hypermodel

    compiled_model = LazyDescribed.from_describable(hypermodel.best_model()) | compile_model(
        **get_baseline_compile_params(strategy=strategy),
    )

    _, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
        compiled_model,
        datasets,
    )

    table.add_row(
        str(validation_metrics['MSE']),
        str(test_metrics['MSE']),
    )
    console.print(table)

    model_evaluator = _cached_model_evaluator('condensation')
    trainable_sizes = []
    total_sizes = []
    losses = []
    for model in hypermodel.iter_best_models():
        compiled_model = LazyDescribed.from_describable(model) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        trainable_size, total_size, validation_loss, _test_loss = model_evaluator(
            compiled_model,
            datasets,
        )

        trainable_sizes.append(trainable_size)
        total_sizes.append(total_size)
        losses.append(validation_loss)

    sns.set_style('whitegrid')

    save_path = _automl_study_path() / f'condensation-each-{each}-trainable.png'
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(trainable_sizes, losses, s=20, color='k')
    ax.scatter(
        baseline_architecture_size,
        baseline_loss,
        facecolors='none',
        edgecolors='k',
        marker='$\\odot$',
        s=100,
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model size')
    ax.set_ylabel('Validation loss')
    f.savefig(str(save_path))

    save_path = _automl_study_path() / f'condensation-each-{each}-total.png'
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(total_sizes, losses, s=20, color='k')
    ax.scatter(
        baseline_architecture_size,
        baseline_loss,
        facecolors='none',
        edgecolors='k',
        marker='$\\odot$',
        s=100,
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model size')
    ax.set_ylabel('Validation loss')
    f.savefig(str(save_path))

    console.print(table)


@functools.cache
def _cached_model_evaluator(
    experiment: Literal['boiling1d', 'condensation'],
    /,
) -> Callable[
    [LazyDescribed[ModelArchitecture], LazyDescribed[ImageDatasetTriplet]],
    tuple[int, int, float, float],
]:
    @cache(JSONAllocator(_model_evaluations_path() / experiment))
    def model_evaluator(
        model: LazyDescribed[ModelArchitecture], datasets: LazyDescribed[ImageDatasetTriplet]
    ) -> tuple[int, int, float, float]:
        _, validation_metrics, test_metrics = evaluate_boiling_model_with_dataset(
            model,
            datasets,
        )

        trainable_size = model().count_parameters(
            trainable=True,
            non_trainable=False,
        )
        total_size = model().count_parameters(
            trainable=True,
            non_trainable=True,
        )

        return (
            trainable_size,
            total_size,
            validation_metrics['MSE'].value,
            test_metrics['MSE'].value,
        )

    return model_evaluator


def _model_evaluations_path() -> Path:
    return _automl_study_path() / 'evaluations'


def _automl_study_path() -> Path:
    return studies_path() / 'automl'
