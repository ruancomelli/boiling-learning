from pathlib import Path

import tensorflow as tf
import typer
from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import BOILING_BASELINE_BATCH_SIZE
from boiling_learning.app.datasets.bridging import to_tensorflow
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_baseline_boiling_architecture,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import datasets_merger, subset

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

    cases_indices = ((0,), (1,), (0, 1), (2,), (3,), (2, 3), (0, 1, 2, 3))

    cross_surface_evaluator = cache(JSONAllocator(_cross_surface_study_path() / 'boiling1d'))(
        _boiling_cross_surface_evaluation
    )

    tables: list[Table] = []
    for metric_name in ('MSE', 'MAPE', 'RMS', 'R2'):
        table = Table(
            'Train \\ Eval',
            *(map(_format_sets, cases_indices)),
            title=f'Cross surface analysis - {metric_name}',
        )

        for training_indices in cases_indices:
            table.add_row(
                _format_sets(training_indices),
                *map(
                    _get_and_format_results,
                    (
                        cross_surface_evaluator(
                            direct_visualization=True,
                            training_cases=training_indices,
                            evaluation_cases=evaluation_cases,
                            strategy=strategy,
                        )[metric_name]
                        for evaluation_cases in cases_indices
                    ),
                    (
                        cross_surface_evaluator(
                            direct_visualization=False,
                            training_cases=training_indices,
                            evaluation_cases=evaluation_cases,
                            strategy=strategy,
                        )[metric_name]
                        for evaluation_cases in cases_indices
                    ),
                ),
                end_section=True,
            )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _boiling_cross_surface_evaluation(
    *,
    direct_visualization: bool,
    training_cases: tuple[int, ...],
    evaluation_cases: tuple[int, ...],
    normalize_images: bool = True,
    strategy: LazyDescribed[tf.distribute.Strategy],
) -> dict[str, float]:
    logger.info(
        f'Training on cases {training_cases} '
        f'| Evaluating on {evaluation_cases} '
        f"| {'Direct' if direct_visualization else 'Indirect'} visualization"
    )

    all_boiling_datasets = boiling_datasets(direct_visualization=direct_visualization)

    if len(training_cases) > 1:
        training_datasets = tuple(
            all_boiling_datasets[training_case] for training_case in training_cases
        )

        training_dataset = LazyDescribed.from_describable(training_datasets) | datasets_merger()
    else:
        (training_case,) = training_cases
        training_dataset = all_boiling_datasets[training_case]

    if len(evaluation_cases) > 1:
        evaluation_datasets = tuple(
            all_boiling_datasets[evaluation_case] for evaluation_case in evaluation_cases
        )
        evaluation_dataset = (
            LazyDescribed.from_describable(evaluation_datasets) | datasets_merger()
        )
    else:
        (evaluation_case,) = evaluation_cases
        evaluation_dataset = all_boiling_datasets[evaluation_case]

    model = get_baseline_boiling_architecture(
        direct_visualization=direct_visualization,
        normalize_images=normalize_images,
        strategy=strategy,
    ) | compile_model(
        get_baseline_compile_params(strategy=strategy),
    )

    logger.info('Training...')

    fit_model = fit_boiling_model(
        model,
        training_dataset,
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
    )

    logger.info('Evaluating')

    model = fit_model.architecture | compile_model(
        get_baseline_compile_params(strategy=strategy),
    )

    ds_evaluation_val = to_tensorflow(
        evaluation_dataset | subset('val'),
        batch_size=BOILING_BASELINE_BATCH_SIZE,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        experiment='boiling1d',
    )

    evaluation = model.evaluate(ds_evaluation_val())

    logger.info(f'Done: {evaluation}')

    return evaluation


def _get_and_format_results(direct_result: float, indirect_result: float) -> str:
    formatted_direct_result = f'[bold]{direct_result:.4f}[/bold]'
    formatted_indirect_result = f'{indirect_result:.4f}'

    ratio = (indirect_result - direct_result) / direct_result
    formatted_ratio = (
        f'[bold][bright_red]{ratio:+.2%}[/bright_red][/bold]'
        if ratio > 0
        else f'[bold][bright_green]{ratio:+.2%}[/bright_green][/bold]'
    )

    return f'{formatted_direct_result}\n{formatted_indirect_result}\n({formatted_ratio})'


def _format_sets(indices: tuple[int, ...]) -> str:
    return ' + '.join(map(str, indices))


def _cross_surface_study_path() -> Path:
    return studies_path() / 'cross-surface'
