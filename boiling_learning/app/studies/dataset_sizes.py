import typer
from rich.console import Console
from rich.table import Table

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.transforms import datasets_merger

app = typer.Typer()
console = Console()

CASES_INDICES = ((0,), (1,), (2,), (3,), (0, 1), (2, 3), (0, 1, 2, 3))
DIRECT_VISUALIZATION = False


@app.command()
def boiling1d() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    table = Table(
        'Dataset',
        'Training length',
        'Validation length',
        'Test length',
        'Total size',
        title='Dataset sizes',
    )

    all_boiling_datasets = boiling_datasets(direct_visualization=DIRECT_VISUALIZATION)
    merged_datasets = {
        indices: _build_merged_datasets(all_boiling_datasets, indices=indices)
        for indices in CASES_INDICES
    }

    for indices, datasets in merged_datasets.items():
        train, val, test = datasets()

        table.add_row(
            f'Dataset {indices}',
            str(len(train)),
            str(len(val)),
            str(len(test)),
            str(len(train) + len(val) + len(test)),
        )

    console.print(table)


@app.command()
def condensation() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    train, val, test = condensation_dataset(each=1)()
    console.print('Train set size:', len(train))
    console.print('Val set size:', len(val))
    console.print('Test set size:', len(test))


def _build_merged_datasets(
    datasets: tuple[LazyDescribed[ImageDatasetTriplet], ...],
    /,
    *,
    indices: tuple[int, ...],
) -> LazyDescribed[ImageDatasetTriplet]:
    if len(indices) > 1:
        selected_datasets = tuple(datasets[training_case] for training_case in indices)
        return LazyDescribed.from_describable(selected_datasets) | datasets_merger()
    else:
        (index,) = indices
        return datasets[index]
