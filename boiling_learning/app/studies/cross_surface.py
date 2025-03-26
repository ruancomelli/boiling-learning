from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import typer
from loguru import logger
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.displaying import glossary, units
from boiling_learning.app.displaying.figures import save_figure
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
from boiling_learning.app.training.evaluation import cached_model_evaluator
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.training import compile_model
from boiling_learning.transforms import datasets_merger
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

METRIC_NAMES = ('MSE', 'MAPE')
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

    tables: dict[tuple[str, str, bool], Table] = {
        (metric_name, subset, direct): Table(
            'Train \\ Eval',
            *(map(_format_sets, CASES_INDICES)),
            title=(
                'Cross surface analysis'
                f' - {metric_name}'
                f' - {subset}'
                f' - {"direct" if direct else "indirect"}'
            ),
        )
        for metric_name in METRIC_NAMES
        for subset in ('training', 'validation', 'test')
        for direct in (False, True)
    }

    cases_latex = {cases: _cases_to_latex(cases) for cases in CASES_INDICES}

    evaluations: list[tuple[str, str, float, float, float, bool, str]] = []
    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'

        for training_indices in CASES_INDICES:
            results = defaultdict[tuple[str, str], list[float]](list)

            for evaluation_indices in CASES_INDICES:
                evaluation = _boiling_cross_surface_evaluation(
                    direct_visualization=direct,
                    training_cases=training_indices,
                    evaluation_cases=evaluation_indices,
                    strategy=strategy,
                )

                for (
                    metric_name,
                    training_metric,
                    validation_metric,
                    test_metric,
                ) in evaluation.iter_metrics():
                    evaluations.append(
                        (
                            cases_latex[training_indices],
                            cases_latex[evaluation_indices],
                            training_metric,
                            validation_metric,
                            test_metric,
                            direct,
                            metric_name,
                        )
                    )
                    if metric_name in METRIC_NAMES:
                        results[metric_name, 'training'].append(training_metric)
                        results[metric_name, 'validation'].append(validation_metric)
                        results[metric_name, 'test'].append(test_metric)

            for metric_name in METRIC_NAMES:
                for subset in 'training', 'validation', 'test':
                    tables[metric_name, subset, direct].add_row(
                        _format_sets(training_indices),
                        *map(str, results[metric_name, subset]),
                        end_section=True,
                    )

    console.print(Columns(tables.values()))

    data = pd.DataFrame(
        evaluations,
        columns=[
            'Training set',
            'Evaluation set',
            'Training metric',
            'Validation metric',
            'Test metric',
            'Visualization mode',
            'Metric name',
        ],
    )
    data['Training set'] = pd.Categorical(data['Training set'], cases_latex.values())
    data['Evaluation set'] = pd.Categorical(data['Evaluation set'], cases_latex.values())

    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'
        for subset, subset_name in zip(
            ('training', 'validation', 'test'),
            ('Training metric', 'Validation metric', 'Test metric'),
        ):
            plot_data = {
                metric_name: (
                    data[
                        (data['Visualization mode'] == direct)
                        & (data['Metric name'] == metric_name)
                    ]
                    .pivot(index='Training set', columns='Evaluation set', values=subset_name)
                    .sort_index(level=0, ascending=True)
                )
                for metric_name in METRIC_NAMES
            }

            f, ax = plt.subplots(1, 1, figsize=(4, 6))

            annot = [
                [f'{mse:.0f}\n\\small({mape:.0f}\\%)' for mse, mape in zip(mse_row, mape_row)]
                for mse_row, mape_row in zip(plot_data['MSE'].values, plot_data['MAPE'].values)
            ]

            sns.heatmap(
                plot_data['MSE'],
                annot=annot,
                norm=LogNorm(),
                cmap=sns.color_palette('Blues', as_cmap=True),
                linewidth=1,
                cbar_kws={
                    'label': f'MSE [{units["mse"]}]',
                    'orientation': 'horizontal',
                    'format': ScalarFormatter(),
                    'pad': 0.065,
                },
                fmt='',
                ax=ax,
            )

            ax.xaxis.tick_top()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
            ax.yaxis.tick_right()
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='left')

            save_figure(
                f,
                _cross_surface_study_path() / f'boiling1d-{direct_label}-{subset}.pdf',
            )
            save_figure(
                f,
                _cross_surface_figures_path() / f'boiling1d-{direct_label}-{subset}.pdf',
            )
            save_figure(
                f,
                _cross_surface_study_path() / f'boiling1d-{direct_label}-{subset}.png',
            )
            save_figure(
                f,
                _cross_surface_figures_path() / f'boiling1d-{direct_label}-{subset}.png',
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

    model = get_baseline_boiling_architecture(
        direct_visualization=direct_visualization,
        normalize_images=True,
        strategy=strategy,
    ) | compile_model(
        **get_baseline_compile_params(strategy=strategy),
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
        **get_baseline_compile_params(strategy=strategy),
    )

    evaluation = cached_model_evaluator('boiling1d')(
        model,
        evaluation_dataset,
        measure_uncertainty=False,
    )

    logger.info(f'Done evaluating: {evaluation}')

    return evaluation


def _format_sets(indices: tuple[int, ...]) -> str:
    return ' + '.join(map(str, indices))


def _cases_to_latex(cases: tuple[int, ...]) -> str:
    if set(cases) == {0, 1}:
        return '$\\overline{\\mathrm{U}}^{\\mathrm{W}}$'
    if set(cases) == {2, 3}:
        return '$\\overline{\\mathrm{U}}^{\\mathrm{R}}$'
    if set(cases) == {0, 1, 2, 3}:
        return '$\\overline{\\mathrm{U}}^{\\mathrm{A}}$'

    assert len(cases) == 1

    return '$' + glossary[CASE_NAMES[cases[0]]] + '$'


def _cross_surface_study_path() -> Path:
    return resolve(studies_path() / 'cross-surface', dir=True)


def _cross_surface_figures_path() -> Path:
    return resolve(figures_path() / 'results' / 'cross-surface', dir=True)
