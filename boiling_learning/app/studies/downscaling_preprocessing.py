from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import typer

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.datasets.raw.condensation import condensation_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.image_datasets import Image
from boiling_learning.preprocessing.image import (
    downscaler,
    grayscaler,
    image_dtype_converter,
    nbins_retained_variance,
    nbins_shannon_entropy_ratio,
)
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


METRICS = (
    (nbins_retained_variance, 'linear'),
    (nbins_shannon_entropy_ratio, 'linear'),
)

DATASET_MARKER_STYLE = (
    ('Thick wire', 'k', 'o'),
    ('Thin wire', 'r', '.'),
    ('Horizontal ribbon', 'g', '>'),
    ('Vertical ribbon', 'b', '^'),
)

INITIAL_PREPROCESSORS = [
    image_dtype_converter('float32'),
    grayscaler(),
]


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    factors: list[int] = typer.Option(list(range(1, 10)) + list(range(10, 50, 10))),
) -> None:
    _plot_downscaling_preprocessing(
        direct,
        output_path=_downscaling_study_path() / 'boiling1d.png',
        factors=factors,
    )


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    sample_frames: list[Image] = []

    for case in condensation_datasets():
        dataset = get_image_dataset(
            case(),
            transformers=INITIAL_PREPROCESSORS,
            experiment='condensation',
        )

        ds_train, _, _ = dataset()
        sample_frame, _ = ds_train[0]
        sample_frames.append(sample_frame)

    sns.set_style('whitegrid')

    NROWS = len(METRICS)
    NCOLS = len(sample_frames)

    f, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(NROWS * 4, NCOLS * 4),
        sharex='col',
        sharey='row',
    )

    factors = [factor**2 for factor in factors]
    preferred_factor = 4
    for col, sample_frame in enumerate(sample_frames):
        downscaled_frames = [downscaler(factor)(sample_frame) for factor in factors]

        for row, (metric, y_scale) in enumerate(METRICS):
            ax = axes[row, col]

            y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

            ax.scatter(factors, y, s=20, color='k')
            ax.scatter(
                factors[0],
                y[0],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )
            ax.scatter(
                factors[preferred_factor],
                y[preferred_factor],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )

            ax.set_xscale('log')
            ax.set_yscale(y_scale)

            if not row:
                ax.set_title(f'Dataset {col}')
            if not col:
                ax.set_ylabel(' '.join(metric.__name__.split('_')).title())

            ax.xaxis.grid(True, which='minor')

    output_path = resolve(
        _downscaling_study_path() / 'condensation.png',
        parents=True,
    )
    f.savefig(str(output_path))


def _plot_downscaling_preprocessing(
    cases,
    /,
    *,
    factors: Sequence[int],
    output_path: Path,
) -> None:
    sns.set_style('whitegrid')

    NROWS = len(METRICS)

    f, axes = plt.subplots(
        NROWS,
        1,
        figsize=(6, NROWS * 4),
        sharex='col',
    )

    for case, (dataset_name, color, marker) in zip(boiling_cases(), DATASET_MARKER_STYLE):
        dataset = get_image_dataset(
            case(),
            transformers=INITIAL_PREPROCESSORS,
            experiment='boiling1d',
        )

        ds_train, _, _ = dataset()
        sample_frame, _ = ds_train[0]
        downscaled_frames = [downscaler(factor)(sample_frame) for factor in factors]

        for row, (metric, y_scale) in enumerate(METRICS):
            ax = axes[row]

            y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

            ax.scatter(factors, y, color=color, marker=marker, label=dataset_name)

            ax.set_xscale('log')
            ax.set_yscale(y_scale)
            ax.legend()

            ax.set_ylabel(' '.join(metric.__name__.split('_')).title())

            ax.xaxis.grid(True, which='minor')

    f.savefig(resolve(output_path, parents=True))


def _downscaling_study_path() -> Path:
    return studies_path() / 'downscaling-preprocessing'
