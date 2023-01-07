from collections.abc import Iterator
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.displaying import units
from boiling_learning.app.displaying.latex import latexify
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
        modin_engine='ray',
        require_gpu=True,
    )

    evaluator = cached_model_evaluator('boiling1d')

    evaluations: dict[tuple[str, bool, str], UncertainValue] = {}
    tables: list[Table] = []
    for direct in False, True:
        model = get_pretrained_baseline_boiling_model(
            direct_visualization=direct,
            normalize_images=True,
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

    with (_validation_study_path() / 'boiling1d-results.txt').open('w') as file:
        file.write(text)


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
    for direct in True, False:
        yield ('\\multicolumn{2}{l}{Direct}' if direct else '\\multicolumn{2}{l}{Indirect}')
        yield '\\\\'  # latex line-break
        for metric_name in evaluation.metrics_names:
            if metric_name == 'loss':
                continue

            yield f'& \\gls{{{metric_name.lower()}}}'
            yield f'& {units[metric_name.lower()]}'

            for subset in 'train', 'val', 'test':
                uncertain_value = evaluations[(metric_name, direct, subset)]
                yield f'& {latexify(uncertain_value.rounded())}'

            if (metric_name, direct, 'val') in _REFERENCE_EVALUATIONS and (
                metric_name,
                direct,
                'test',
            ) in _REFERENCE_EVALUATIONS:
                yield '&'
                yield f'& {latexify(_REFERENCE_EVALUATIONS[(metric_name, direct, "val")])}'
                yield f'& {latexify(_REFERENCE_EVALUATIONS[(metric_name, direct, "test")])}'
            yield '\\\\'


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    model = get_pretrained_baseline_condensation_model(
        each=each,
        normalize_images=normalize,
        strategy=strategy,
    )

    console.print(model.validation_metrics)
    console.print(model.test_metrics)


def _validation_study_path() -> Path:
    return resolve(studies_path() / 'validation', dir=True)
