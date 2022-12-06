import matplotlib.pyplot as plt
import seaborn as sns
import typer
from skimage.metrics import normalized_mutual_information as nmi

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.image_datasets import Image
from boiling_learning.preprocessing.image import (
    downscaler,
    normalized_mutual_information,
    retained_variance,
    shannon_cross_entropy_ratio,
    shannon_entropy_ratio,
    structural_similarity_ratio,
)

app = typer.Typer()

BOILING_DOWNSCALING_INDEX = 3
CONDENSATION_DOWNSCALING_INDEX = 3

METRICS = (
    retained_variance,
    shannon_cross_entropy_ratio,
    shannon_entropy_ratio,
    structural_similarity_ratio,
    normalized_mutual_information,
    nmi,
)


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    """Validate current implementation against reference."""
    sample_frames: list[Image] = []

    preprocessors = default_boiling_preprocessors(direct_visualization=direct)[
        :BOILING_DOWNSCALING_INDEX
    ]

    for case in boiling_cases():
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
        )

        ds_train, _, _ = dataset()
        sample_frame, _ = ds_train[0]
        sample_frames.append(sample_frame)

    sns.set_style('whitegrid')

    f, axes = plt.subplots(
        len(METRICS),
        len(sample_frames),
        figsize=(16, 16),
        sharex='row',
        sharey='col',
    )

    x = factors
    preferred_factor = 4
    for col, sample_frame in enumerate(sample_frames):
        downscaled_frames = [downscaler(factor)(sample_frame) for factor in factors]

        for row, metric in enumerate(METRICS):
            ax = axes[row, col]

            y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

            ax.scatter(x, y, s=20, color='k')
            ax.scatter(
                x[0],
                y[0],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )
            ax.scatter(
                x[preferred_factor],
                y[preferred_factor],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )

            if not row:
                ax.set_title(f'Dataset {col}')
            if not col:
                ax.set_ylabel(' '.join(metric.__name__.split('_')).title())

            ax.xaxis.grid(True, which='minor')

    f.savefig('tmp.png')


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    raise NotImplementedError
