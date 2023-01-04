import itertools

import typer
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.boiling1d import get_pretrained_baseline_boiling_model
from boiling_learning.app.training.common import get_baseline_compile_params
from boiling_learning.app.training.evaluation import cached_model_evaluator
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

    evaluator = cached_model_evaluator('boiling1d')
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
            table.add_row(
                metric,
                str(evaluation.training_metrics[metric]),
                str(evaluation.validation_metrics[metric]),
                str(evaluation.test_metrics[metric]),
            )

        tables.append(table)

    console.print(Columns(tables))


@app.command()
def condensation() -> None:
    raise NotImplementedError
