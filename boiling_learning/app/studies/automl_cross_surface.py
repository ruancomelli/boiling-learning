from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import typer
from loguru import logger
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from rich.console import Console
from rich.table import Table

from boiling_learning.app.automl.autofit_dataset import best_baseline_boiling1d_model
from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.displaying import glossary, units
from boiling_learning.app.displaying.figures import save_figure
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    fit_boiling_model,
    get_baseline_model_size,
)
from boiling_learning.app.training.common import (
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import datasets_merger
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

CASES_INDICES = ((0,), (1,), (2,), (3,), (0, 1), (2, 3), (0, 1, 2, 3))
CASE_NAMES = {
    0: 'large wire ds',
    1: 'small wire ds',
    2: 'horizontal ribbon ds',
    3: 'vertical ribbon ds',
}


@app.command()
@logger.catch
def boiling1d() -> None:
    strategy = configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    table = Table(
        'Train \\ Eval',
        *(map(_format_sets, CASES_INDICES)),
        title='Cross surface analysis',
    )

    cases_latex = {cases: _cases_to_latex(cases) for cases in CASES_INDICES}

    evaluations: list[tuple[str, str, float, bool]] = []
    for training_indices in CASES_INDICES:
        formated_results: list[str] = []

        for evaluation_indices in CASES_INDICES:
            direct_visualization_evaluation = _boiling_cross_surface_evaluation(
                direct_visualization=True,
                training_cases=training_indices,
                evaluation_cases=evaluation_indices,
                strategy=strategy,
            ).validation_metrics['MSE']
            indirect_visualization_evaluation = _boiling_cross_surface_evaluation(
                direct_visualization=False,
                training_cases=training_indices,
                evaluation_cases=evaluation_indices,
                strategy=strategy,
            ).validation_metrics['MSE']

            evaluations.extend(
                (
                    (
                        cases_latex[training_indices],
                        cases_latex[evaluation_indices],
                        direct_visualization_evaluation,
                        True,
                    ),
                    (
                        cases_latex[training_indices],
                        cases_latex[evaluation_indices],
                        indirect_visualization_evaluation,
                        False,
                    ),
                )
            )

            formated_results.append(
                _get_and_format_results(
                    direct_visualization_evaluation,
                    indirect_visualization_evaluation,
                )
            )

        table.add_row(
            _format_sets(training_indices),
            *formated_results,
            end_section=True,
        )

    console.print(table)

    data = pd.DataFrame(
        evaluations,
        columns=['Training set', 'Evaluation set', 'Result', 'Visualization'],
    )
    data['Training set'] = pd.Categorical(data['Training set'], cases_latex.values())
    data['Evaluation set'] = pd.Categorical(data['Evaluation set'], cases_latex.values())

    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'

        plot_data = (
            data[data['Visualization'] == direct]
            .pivot(index='Training set', columns='Evaluation set', values='Result')
            .sort_index(level=0, ascending=True)
        )

        f, ax = plt.subplots(1, 1, figsize=(5, 5))

        sns.heatmap(
            plot_data,
            annot=True,
            norm=LogNorm(),
            cmap=sns.color_palette('Blues', as_cmap=True),
            linewidth=1,
            cbar_kws={
                'label': f'Validation loss [{units["mse"]}]',
                'orientation': 'horizontal',
                'format': ScalarFormatter(),
            },
            fmt='.0f',
            ax=ax,
        )
        ax.xaxis.tick_top()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
        ax.yaxis.tick_right()
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='left')

        save_figure(
            f,
            _automl_cross_surface_study_path() / f'boiling1d-{direct_label}.pdf',
        )
        save_figure(
            f,
            _automl_cross_surface_figures_path() / f'boiling1d-{direct_label}.pdf',
        )


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _boiling_cross_surface_evaluation(
    *,
    direct_visualization: bool,
    training_cases: tuple[int, ...],
    evaluation_cases: tuple[int, ...],
    strategy: LazyDescribed[tf.distribute.Strategy],
):
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

    model = best_baseline_boiling1d_model(
        direct_visualization=direct_visualization,
        strategy=strategy,
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        normalize_images=True,
        max_model_size=get_baseline_model_size(
            direct_visualization=direct_visualization,
            strategy=strategy,
        ),
    ) | compile_model(
        **get_baseline_compile_params(strategy=strategy),
    )

    logger.info('Training...')

    fit_model = fit_boiling_model(
        model,
        training_dataset,
        # use batch size of `32` because this is how automl models are trained
        get_baseline_fit_params(batch_size=32),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
    )

    logger.info('Evaluating')

    model = fit_model.architecture | compile_model(
        **get_baseline_compile_params(strategy=strategy),
    )

    evaluation = cached_model_evaluator('boiling1d')(
        model,
        evaluation_dataset,
        measure_uncertainty=False,
    )

    logger.info(f'Done evaluating: {evaluation}')

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


def _cases_to_latex(cases: tuple[int, ...]) -> str:
    return (
        (
            '${\\mkern 1.5mu\\overline{\\mkern-1.5mu\\bigcup\\mkern-1.5mu}\\mkern 1.5mu}\\left('
            + ', '.join(glossary[CASE_NAMES[case]] for case in cases)
            + '\\right)$'
        )
        if len(cases) > 1
        else '$' + glossary[CASE_NAMES[cases[0]]] + '$'
    )


def _automl_cross_surface_study_path() -> Path:
    return resolve(studies_path() / 'automl-cross-surface', dir=True)


def _automl_cross_surface_figures_path() -> Path:
    return resolve(figures_path() / 'results' / 'automl-cross-surface', dir=True)
