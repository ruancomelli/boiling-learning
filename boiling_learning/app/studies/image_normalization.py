import itertools
from collections.abc import Iterator
from pathlib import Path

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.displaying.latex import latexify
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import get_pretrained_baseline_boiling_model
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import ModelEvaluation, cached_model_evaluator
from boiling_learning.model.evaluate import UncertainValue
from boiling_learning.model.training import compile_model
from boiling_learning.utils.pathutils import resolve

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

    evaluator = cached_model_evaluator('boiling1d')
    evaluations: dict[tuple[bool, bool, str, str], UncertainValue] = {}
    tables: list[Table] = []
    for normalize, direct_visualization in itertools.product((False, True), repeat=2):
        table = Table(
            'Metric',
            'Training',
            'Validation',
            'Test',
            title=(
                ('Normalized' if normalize else 'Non-normalized')
                + ' - '
                + ('Direct' if direct_visualization else 'Indirect')
            ),
        )

        model = get_pretrained_baseline_boiling_model(
            direct_visualization=direct_visualization,
            normalize_images=normalize,
            strategy=strategy,
        )

        compiled_model = model.architecture | compile_model(
            **get_baseline_compile_params(strategy=strategy),
        )

        evaluation_dataset = baseline_boiling_dataset(direct_visualization=direct_visualization)

        evaluation = evaluator(
            compiled_model,
            evaluation_dataset,
        )

        for metric in evaluation.metrics_names:
            evaluations[
                (normalize, direct_visualization, metric, 'train')
            ] = evaluation.training_metrics[metric]
            evaluations[
                (normalize, direct_visualization, metric, 'val')
            ] = evaluation.validation_metrics[metric]
            evaluations[
                (normalize, direct_visualization, metric, 'test')
            ] = evaluation.test_metrics[metric]
            table.add_row(
                metric,
                str(evaluation.training_metrics[metric]),
                str(evaluation.validation_metrics[metric]),
                str(evaluation.test_metrics[metric]),
            )

        tables.append(table)

    console.print(Columns(tables))

    text = _build_latex_table(evaluations, evaluation)

    with (_image_normalization_study_path() / 'results.txt').open('w') as file:
        file.write(text)


def _build_latex_table(
    evaluations: dict[tuple[bool, bool, str, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> str:
    return '\n'.join(_latex_table_lines(evaluations, evaluation))


def _latex_table_lines(
    evaluations: dict[tuple[bool, bool, str, str], UncertainValue],
    evaluation: ModelEvaluation,
) -> Iterator[str]:
    for normalized in False, True:
        yield (
            '\\multicolumn{2}{l}{Normalized}'
            if normalized
            else '\\multicolumn{2}{l}{Non-normalized}'
        )
        yield '\\\\'  # latex line-break
        for metric_name in evaluation.metrics_names:
            yield f'& \\gls{{{metric_name.lower()}}}'

            for direct_visualization in True, None, False:
                if direct_visualization is None:
                    yield '&'
                else:
                    for subset in 'train', 'val', 'test':
                        uncertain_value = evaluations[
                            (normalized, direct_visualization, metric_name, subset)
                        ]
                        rounded_uncertain_value = uncertain_value.rounded()
                        yield f'& {latexify(rounded_uncertain_value)}'
            yield '\\\\'


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _image_normalization_study_path() -> Path:
    return resolve(studies_path() / 'image-normalization', dir=True)
