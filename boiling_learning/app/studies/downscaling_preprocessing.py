from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.displaying.figures import save_figure
from boiling_learning.app.paths import studies_path
from boiling_learning.image_datasets import Image, ImageDataset
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.image import (
    downscaler,
    nbins_retained_variance,
    nbins_shannon_entropy_ratio,
)
from boiling_learning.transforms import subset
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


METRICS = (
    (nbins_retained_variance, 'retained-variance', 'Relative variance'),
    (nbins_shannon_entropy_ratio, 'cross-entropy', 'Cross-entropy ratio'),
)

DATASET_MARKER_STYLE = (
    ('Large wire', 'k', 'o'),
    ('Small wire', 'r', '.'),
    ('Horizontal ribbon', 'g', '>'),
    ('Vertical ribbon', 'b', '^'),
)

DEFAULT_FACTORS = list(range(1, 11))


@app.command()
def boiling1d(factors: list[int] = typer.Option(DEFAULT_FACTORS)) -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    factors = sorted(frozenset(factors) | {1})

    for metric, metric_id, metric_name in METRICS:
        described_metric = LazyDescribed.from_value_and_description(metric, metric_id)
        data = _get_data(factors, described_metric)

        f, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))

        sns.scatterplot(
            data,
            ax=ax,
            x='Downscaling factor',
            y='Metric',
            hue='Dataset',
            style='Dataset',
            markers=[marker for _, _, marker in DATASET_MARKER_STYLE],
            alpha=0.75,
        )
        ax.set(
            xticks=factors,
            xticklabels=[str(factor) for factor in factors],
            ylabel=metric_name,
        )

        save_figure(f, _downscaling_study_path() / f'boiling1d-{metric_id}.pdf')
        save_figure(f, _downscaling_figures_path() / f'{metric_id}.pdf')


def _get_data(
    factors: list[int],
    metric: LazyDescribed[Callable[[Image, Image], float]],
) -> pd.DataFrame:
    @cache(JSONAllocator(_downscaling_study_path() / 'per-frame'))
    def _per_frame_data_getter(
        dataset: LazyDescribed[ImageDataset],
        /,
        *,
        index: int,
        metric: LazyDescribed[Callable[[Image, Image], float]],
        factor: int,
    ) -> float:
        frame, _ = dataset()[index]
        downscaled = downscaler(factor)(frame)
        return metric()(frame, downscaled)

    data = []
    for case, (dataset_name, _color, _marker) in zip(boiling_cases(), DATASET_MARKER_STYLE):
        for factor in factors:
            preprocessors = default_boiling_preprocessors(
                direct_visualization=True,
                downscale_factor=factor,
            )
            preprocessors = [preprocessors[0], preprocessors[1][:1]]  # remove cropping steps
            datasets = get_image_dataset(
                case(),
                transformers=preprocessors,
                experiment='boiling1d',
                shuffle=False,
            )
            for subset_name in ('train',):
                # for subset_name in 'train', 'val', 'test':
                dataset = datasets | subset(subset_name)

                for index in [0]:
                    # for index in range(len(dataset())):
                    result = _per_frame_data_getter(
                        dataset,
                        index=index,
                        metric=metric,
                        factor=factor,
                    )
                    data.append((factor, result, dataset_name))

    return pd.DataFrame(data, columns=['Downscaling factor', 'Metric', 'Dataset'])


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    raise NotImplementedError


def _downscaling_figures_path() -> Path:
    return resolve(
        figures_path() / 'machine-learning' / 'preprocessing' / 'downscaling',
        dir=True,
    )


def _downscaling_study_path() -> Path:
    return resolve(studies_path() / 'downscaling-preprocessing', dir=True)
