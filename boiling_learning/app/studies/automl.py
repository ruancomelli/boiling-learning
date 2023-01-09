from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import autofit_dataset
from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    get_pretrained_baseline_boiling_model,
)
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import dataset_sampler
from boiling_learning.utils.pathutils import resolve

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
    model_evaluator = cached_model_evaluator('boiling1d')

    tables: list[Table] = []
    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'

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
        )

        compiled_model = LazyDescribed.from_describable(hypermodel.best_model()) | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )
        evaluation = model_evaluator(compiled_model, datasets)

        best_model_size_table = Table(
            'Size',
            'Value',
            title=f'AutoML best model sizes - {direct_label} visualization',
        )
        best_model_size_table.add_row('Trainable', str(evaluation.trainable_parameters_count))
        best_model_size_table.add_row('Total', str(evaluation.total_parameters_count))
        best_model_size_table.add_row(
            'Relative trainable',
            str(evaluation.trainable_parameters_count / baseline_architecture_size),
        )
        tables.append(best_model_size_table)

        best_model_table = Table(
            'Metric',
            'Training',
            'Validation',
            'Test',
            title=f'AutoML best model metrics - {direct_label} visualization',
        )
        for metric in evaluation.metrics_names:
            if metric == 'loss':
                continue

            best_model_table.add_row(
                metric,
                str(evaluation.training_metrics[metric]),
                str(evaluation.validation_metrics[metric]),
                str(evaluation.test_metrics[metric]),
            )
        tables.append(best_model_table)

        trainable_sizes = []
        total_sizes = []
        losses = []
        for model in hypermodel.iter_best_models():
            compiled_model = LazyDescribed.from_describable(model) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            evaluation = model_evaluator(compiled_model, datasets)

            trainable_sizes.append(evaluation.trainable_parameters_count)
            total_sizes.append(evaluation.total_parameters_count)
            losses.append(evaluation.validation_metrics['loss'])

        save_path = _automl_study_path() / f'boiling1d-{direct_label}-each-{each}-trainable.pdf'
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
            / f"boiling1d-{'direct' if direct else 'indirect'}-each-{each}-total.pdf"
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
    raise NotImplementedError


def _automl_study_path() -> Path:
    return resolve(studies_path() / 'automl', dir=True)
