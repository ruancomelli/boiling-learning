import textwrap
from collections.abc import Iterator
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.latex import NEW_LINE_TOKEN, latexify
from boiling_learning.app.figures.architectures import diagrams_path
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import get_pretrained_baseline_boiling_model
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.condensation import get_pretrained_baseline_condensation_model
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

    evaluations: dict[tuple[str, bool, str], UncertainValue] = {}
    tables: list[Table] = []
    for direct in False, True:
        model = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=False,
            strategy=strategy,
        )

        compiled_model = model.architecture | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        evaluation_dataset = baseline_boiling_dataset(direct_visualization=direct)

        evaluation = evaluator(
            compiled_model,
            evaluation_dataset,
        )

        table = Table(
            'Metric',
            'Training',
            'Validation',
            'Test',
            title=(('Direct' if direct else 'Indirect') + ' validation'),
        )
        for metric in evaluation.metrics_names:
            table.add_row(
                metric,
                str(evaluation.training_metrics[metric]),
                str(evaluation.validation_metrics[metric]),
                str(evaluation.test_metrics[metric]),
            )
            evaluations[(metric, direct, 'train')] = evaluation.training_metrics[metric]
            evaluations[(metric, direct, 'val')] = evaluation.validation_metrics[metric]
            evaluations[(metric, direct, 'test')] = evaluation.test_metrics[metric]

        tables.append(table)

    console.print(Columns(tables))

    text = _build_latex_table(evaluations, evaluation)
    (_validation_study_path() / 'boiling1d-results.tex').write_text(text)
    (_validation_study_results_path() / 'validate.tex').write_text(text)


_REFERENCE_EVALUATIONS = {
    ('r2', True, 'val'): UncertainValue(0.9828, upper=0.0017, lower=0.0019),
    ('r2', True, 'test'): UncertainValue(0.9826, upper=0.0018, lower=0.0020),
    ('r2', False, 'val'): UncertainValue(0.9557, upper=0.0046, lower=0.0050),
    ('r2', False, 'test'): UncertainValue(0.9564, upper=0.0049, lower=0.0050),
    ('mape', True, 'val'): UncertainValue(7.37, upper=0.46, lower=0.42),
    ('mape', True, 'test'): UncertainValue(7.62, upper=0.44, lower=0.42),
    ('mape', False, 'val'): UncertainValue(10.60, upper=0.56, lower=0.56),
    ('mape', False, 'test'): UncertainValue(10.35, upper=0.52, lower=0.51),
    ('mae', True, 'val'): UncertainValue(2.77, upper=0.14, lower=0.14),
    ('mae', True, 'test'): UncertainValue(2.66, upper=5.78, lower=2.55),
    ('mae', False, 'val'): UncertainValue(3.98, upper=9.93, lower=3.85),
    ('mae', False, 'test'): UncertainValue(3.97, upper=9.82, lower=3.83),
    ('mse', True, 'val'): UncertainValue(13.03, upper=1.37, lower=1.19),
    ('mse', True, 'test'): UncertainValue(13.19, upper=1.49, lower=1.31),
    ('mse', False, 'val'): UncertainValue(33.55, upper=4.05, lower=3.73),
    ('mse', False, 'test'): UncertainValue(32.87, upper=3.92, lower=3.65),
}


def _build_latex_table(
    evaluations: dict[tuple[str, bool, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> str:
    return '\n'.join(_latex_table_lines(evaluations, evaluation))


def _latex_table_lines(
    evaluations: dict[tuple[str, bool, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> Iterator[str]:
    yield textwrap.dedent(
        '''
        \\begin{tabular}{@{}lrllllcll@{}}\\toprule
            &
            &
            & \\multicolumn{3}{c}{This work}
            &
            & \\multicolumn{2}{c}{Reference}
            \\\\ \\cmidrule{4-6} \\cmidrule{8-9}
            & Metric
            & Unit
            & Training
            & Validation
            & Test
            &
            & Validation
            & Test
            \\\\
            \\midrule
        '''
    )

    evaluations = {
        (metric_name.lower(), direct, subset): value
        for (metric_name, direct, subset), value in evaluations.items()
    }

    for direct in True, False:
        yield ('\\multicolumn{2}{l}{Direct}' if direct else '\\multicolumn{2}{l}{Indirect}')
        yield NEW_LINE_TOKEN
        for metric_name in map(str.lower, evaluation.metrics_names):
            if metric_name == 'loss':
                continue

            yield f'& {_metric_name_to_gls(metric_name)}'
            yield f'& {units[metric_name]}'

            for subset in 'train', 'val', 'test':
                uncertain_value = evaluations[(metric_name, direct, subset)]
                yield f'& {latexify(uncertain_value)}'

            yield '&'
            yield f'& {latexify(_REFERENCE_EVALUATIONS[(metric_name, direct, "val")])}' if (
                metric_name,
                direct,
                'val',
            ) in _REFERENCE_EVALUATIONS else '& ---'
            yield f'& {latexify(_REFERENCE_EVALUATIONS[(metric_name, direct, "test")])}' if (
                metric_name,
                direct,
                'test',
            ) in _REFERENCE_EVALUATIONS else '& ---'

            yield NEW_LINE_TOKEN

    yield '\\bottomrule'
    yield '\\end{tabular}'


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    model = get_pretrained_baseline_condensation_model(
        each=each,
        normalize_images=normalize,
        strategy=strategy,
    )

    console.print(model.validation_metrics)
    console.print(model.test_metrics)


def _metric_name_to_gls(metric_name: str) -> str:
    return '\\gls{rmse}' if metric_name == 'rms' else f'\\gls{{{metric_name}}}'


def _validation_study_results_path() -> Path:
    return resolve(diagrams_path() / 'machine-learning', dir=True)


def _validation_study_path() -> Path:
    return resolve(studies_path() / 'validate', dir=True)
