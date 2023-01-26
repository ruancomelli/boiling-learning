import textwrap
from collections.abc import Iterator
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.figures import DATASET_MARKER_STYLE
from boiling_learning.app.displaying.latex import NEW_LINE_TOKEN
from boiling_learning.app.figures.architectures import diagrams_path
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import fit_boiling_model
from boiling_learning.app.training.common import (
    get_baseline_architecture,
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import ModelEvaluation, cached_model_evaluator
from boiling_learning.model.evaluate import UncertainValue
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()


@app.command()
def boiling1d() -> None:
    """Validate current implementation against reference."""
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    evaluator = cached_model_evaluator('boiling1d')

    evaluations: dict[tuple[str, str, bool, str], UncertainValue] = {}
    tables: list[Table] = []

    for direct in False, True:
        for (dataset_name, _), dataset in zip(
            DATASET_MARKER_STYLE,
            boiling_datasets(direct_visualization=direct),
        ):
            model = get_baseline_architecture(
                dataset,
                normalize_images=True,
                strategy=strategy,
            ) | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            fit_model = fit_boiling_model(
                model,
                dataset,
                get_baseline_fit_params(),
                strategy=strategy,
            )

            compiled_model = fit_model.architecture | compile_model(
                **get_baseline_compile_params(strategy=strategy),
            )

            evaluation = evaluator(compiled_model, dataset, measure_uncertainty=False)

            table = Table(
                'Metric',
                'Training',
                'Validation',
                'Test',
                title=f'{dataset_name} - {"direct" if direct else "indirect"}',
            )
            for metric in evaluation.metrics_names:
                table.add_row(
                    metric,
                    str(evaluation.training_metrics[metric]),
                    str(evaluation.validation_metrics[metric]),
                    str(evaluation.test_metrics[metric]),
                )
                evaluations[(dataset_name, metric, direct, 'train')] = evaluation.training_metrics[
                    metric
                ]
                evaluations[(dataset_name, metric, direct, 'val')] = evaluation.validation_metrics[
                    metric
                ]
                evaluations[(dataset_name, metric, direct, 'test')] = evaluation.test_metrics[
                    metric
                ]

            tables.append(table)

    console.print(Columns(tables))

    text = _build_latex_table(evaluations, evaluation)
    (_single_surface_study_path() / 'single-surface.tex').write_text(text)
    (_single_surface_study_results_path() / 'single-surface.tex').write_text(text)


def _build_latex_table(
    evaluations: dict[tuple[str, str, bool, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> str:
    return '\n'.join(_latex_table_lines(evaluations, evaluation))


def _latex_table_lines(
    evaluations: dict[tuple[str, str, bool, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> Iterator[str]:
    yield textwrap.dedent(
        '''
        \\begin{tabular}{@{}lrllllclll@{}}\\toprule
            &
            &
            & \\multicolumn{3}{c}{Direct visualization}
            &
            & \\multicolumn{3}{c}{Indirect visualization}
            \\\\ \\cmidrule{4-6} \\cmidrule{8-10}
            & Metric
            & Unit
            & Training
            & Validation
            & Test
            &
            & Training
            & Validation
            & Test
            \\\\
            \\midrule
        '''
    )

    evaluations = {
        (dataset_name, metric_name.lower(), direct, subset): value
        for (dataset_name, metric_name, direct, subset), value in evaluations.items()
    }

    for dataset_name, _ in DATASET_MARKER_STYLE:
        yield f'\\multicolumn{{2}}{{l}}{{{dataset_name}}}'
        yield NEW_LINE_TOKEN
        for metric_name in map(str.lower, evaluation.metrics_names):
            if metric_name == 'loss':
                continue

            yield f'& {_metric_name_to_gls(metric_name)}'
            yield f'& {units[metric_name]}'

            for subset in 'train', 'val', 'test':
                value = evaluations[(dataset_name, metric_name, True, subset)]
                yield f'& {value:.2f}'

            yield '&'

            for subset in 'train', 'val', 'test':
                value = evaluations[(dataset_name, metric_name, False, subset)]
                yield f'& {value:.2f}'

            yield NEW_LINE_TOKEN

    yield '\\bottomrule'
    yield '\\end{tabular}'


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _metric_name_to_gls(metric_name: str) -> str:
    return '\\gls{rmse}' if metric_name == 'rms' else f'\\gls{{{metric_name}}}'


def _single_surface_study_results_path() -> Path:
    return resolve(diagrams_path() / 'results', dir=True)


def _single_surface_study_path() -> Path:
    return resolve(studies_path() / 'single-surface', dir=True)
