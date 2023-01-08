from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rich
import seaborn as sns
import typer

from boiling_learning.app.configuration import configure
from boiling_learning.app.datasets.bridged.boiling1d import DEFAULT_BOILING_OUTLIER_FILTER
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.app.training.boiling1d import DEFAULT_BOILING_HEAT_FLUX_TARGET
from boiling_learning.datasets.sliceable import targets
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = rich.console.Console()


@app.command()
def boiling1d() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    datasets = boiling_datasets(direct_visualization=True)
    f, axes = plt.subplots(len(datasets), 1, figsize=(6, 4))
    for index, (ax, dataset) in enumerate(zip(axes, datasets)):
        data = _sorted_boiling_datasets(dataset)

        sns.scatterplot(
            ax=ax,
            data=data,
            x='index',
            y='heat flux',
            hue='class',
            alpha=0.5,
        )
        ax.set_title(f'Dataset {index}')

    f.savefig(str(_data_split_study_path() / 'boiling1d.pdf'))


@app.command()
def condensation(
    each: int = typer.Option(60),
    normalize: bool = typer.Option(...),
) -> None:
    raise NotImplementedError


def _sorted_boiling_datasets(datasets: LazyDescribed[ImageDatasetTriplet]) -> pd.DataFrame:
    ds_train, ds_val, ds_test = datasets()

    df = pd.DataFrame(
        sorted(
            (
                (
                    target['nominal_power'],
                    target[DEFAULT_BOILING_HEAT_FLUX_TARGET],
                    target['elapsed_time'],
                    class_name,
                )
                for class_name, ds in (
                    ('train', ds_train),
                    ('val', ds_val),
                    ('test', ds_test),
                )
                for target in targets(ds).prefetch(1024)
                if DEFAULT_BOILING_OUTLIER_FILTER()(None, target)
            ),
            key=lambda power_hf_et_class: (
                power_hf_et_class[0],
                power_hf_et_class[2],
            ),
        ),
        columns=['nominal power', 'heat flux', 'elapsed time', 'class'],
    )
    df['index'] = range(len(df))
    return df


def _data_split_study_path() -> Path:
    return resolve(studies_path() / 'data-split', dir=True)
