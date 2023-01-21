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
from boiling_learning.app.displaying.figures import DATASET_MARKER_STYLE, save_figure
from boiling_learning.app.paths import studies_path
from boiling_learning.datasets.sliceable import features
from boiling_learning.preprocessing.image import structural_similarity
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


DEFAULT_INDICES = list(range(100))


@app.command()
def boiling1d(indices: list[int] = typer.Option(DEFAULT_INDICES)) -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    indices = sorted(frozenset(indices) | {0})

    metrics: list[tuple[int, float, str]] = []
    for case, (dataset_name, _) in zip(boiling_cases(), DATASET_MARKER_STYLE):
        preprocessors = [
            default_boiling_preprocessors(
                direct_visualization=False,
                downscale_factor=1,
            )[0]
        ]
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
            shuffle=False,
        )

        ds_train, _, _ = dataset()
        frames = features(ds_train).fetch(indices)
        reference_frame = frames[0]
        metrics.extend(
            (index, structural_similarity(reference_frame, frame), dataset_name)
            for index, frame in zip(indices, frames)
        )

    df = pd.DataFrame(
        metrics,
        columns=['Index', 'Structural similarity ratio', 'Dataset'],
    )

    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    sns.scatterplot(
        ax=ax,
        data=df,
        x='Index',
        y='Structural similarity ratio',
        hue='Dataset',
        style='Dataset',
        markers=[marker for _, marker in DATASET_MARKER_STYLE],
        alpha=0.5,
    )
    ax.set(xscale='linear', yscale='linear', ylim=(0, 1))

    save_figure(f, _consecutive_frames_study_path() / 'boiling1d.pdf')
    save_figure(f, _consecutive_frames_figures_path() / 'boiling1d.pdf')


@app.command()
def condensation(
    indices: list[int] = typer.Option(DEFAULT_INDICES),
) -> None:
    raise NotImplementedError


def _consecutive_frames_figures_path() -> Path:
    return resolve(
        figures_path() / 'machine-learning' / 'preprocessing' / 'consecutive-frames',
        dir=True,
    )


def _consecutive_frames_study_path() -> Path:
    return resolve(studies_path() / 'consecutive-frames', dir=True)
