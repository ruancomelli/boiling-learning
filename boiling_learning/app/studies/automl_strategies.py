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
from boiling_learning.app.training.evaluation import (
    cached_model_evaluator,
    evaluate_boiling_model_with_dataset,
)
from boiling_learning.automl.tuners import (
    EarlyStoppingBayesian,
    EarlyStoppingGreedy,
    EarlyStoppingHyperband,
)
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model

app = typer.Typer()
console = Console()


@app.command()
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    tables: list[Table] = []
    for direct in False, True:
        for tuner_class in EarlyStoppingGreedy, EarlyStoppingHyperband, EarlyStoppingBayesian:
            table = Table(
                'Validation loss',
                'Test loss',
                title=(
                    'Automatic machine learning - '
                    + ('direct' if direct else 'indirect')
                    + ' visualization'
                    + f' {tuner_class.__name__}'
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

            hypermodel = autofit_dataset(
                datasets,
                target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
                normalize_images=True,
                max_model_size=baseline_architecture_size,
                goal=None,
                experiment='boiling1d',
                strategy=strategy,
                tuner_class=tuner_class,
            )

            # TODO: fix this!!
            compiled_model = LazyDescribed.from_describable(
                hypermodel.best_model()
            ) | compile_model(
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

            model_evaluator = cached_model_evaluator('boiling1d')
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

            save_path = (
                _automl_strategy_study_path()
                / f"boiling1d-{'direct' if direct else 'indirect'}-trainable.png"
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
                _automl_strategy_study_path()
                / f"boiling1d-{'direct' if direct else 'indirect'}-total.png"
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
def condensation(each: int = typer.Option(1)) -> None:
    pass


def _automl_strategy_study_path() -> Path:
    return studies_path() / 'automl-strategy'
