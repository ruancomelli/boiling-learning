from fractions import Fraction
from pathlib import Path

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
from boiling_learning.app.training.evaluation import evaluate_boiling_model_with_dataset
from boiling_learning.lazy import LazyDescribed
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
    for direct in (True,):
        # for direct in (False, True):
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

        # baseline_loss = baseline_fit_return.validation_metrics['MSE']
        baseline_architecture_size = baseline_fit_return.architecture().count_parameters(
            trainable=True,
            non_trainable=False,
        )

        datasets = baseline_boiling_dataset(direct_visualization=direct)

        if each != 1:
            datasets = datasets | dataset_sampler(Fraction(1, each), subset='train')

        autofit_result = autofit_dataset(
            datasets,
            target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
            normalize_images=True,
            max_model_size=baseline_architecture_size,
            goal=None,
            experiment='boiling1d',
            strategy=strategy,
        )

        compiled_model = LazyDescribed.from_describable(
            autofit_result.hypermodel.best_model()
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

        # save_path = _automl_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png"

        # sizes = []
        # losses = []
        # for model in autofit_result.hypermodel.iter_best_models():
        #     compiled_model = LazyDescribed.from_describable(model) | compile_model(
        #         **get_baseline_compile_params(strategy=strategy),
        #     )

        #     _, validation_metrics, _ = evaluate_boiling_model_with_dataset(
        #         compiled_model,
        #         datasets,
        #     )

        #     model_size = model.count_parameters(
        #         trainable=True,
        #         non_trainable=False,
        #     )

        #     sizes.append(model_size)
        #     losses.append(validation_metrics["MSE"])

        # sns.set_style('whitegrid')

        # f, ax = plt.subplots(
        #     1,
        #     1,
        #     figsize=(6, 4),
        #     sharex='col',
        #     sharey='row',
        # )
        # ax.scatter(sizes, losses, s=20, color='k')
        # ax.scatter(
        #     baseline_architecture_size,
        #     baseline_loss,
        #     facecolors='none',
        #     edgecolors='k',
        #     marker='$\\odot$',
        #     s=100,
        # )
        # f.savefig(str(save_path))

    console.print(Columns(tables))


@app.command()
def condensation(normalize: bool = typer.Option(...)) -> None:
    raise NotImplementedError


def _automl_study_path() -> Path:
    return studies_path() / 'automl'
