from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rich
import seaborn as sns
import typer

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.bridged.boiling1d import DEFAULT_BOILING_OUTLIER_FILTER
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.displaying import glossary, units
from boiling_learning.app.paths import studies_path
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.utils.pathutils import PathLike, resolve

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

    for direct in False, True:
        data = _cached_data_getter()(direct_visualization=direct)

        f, ax = plt.subplots(1, 1, figsize=(12, 4))
        sns.boxenplot(
            ax=ax,
            data=data,
            x='nominal power',
            y='brightness',
            hue='Subset',
            hue_order=['Training', 'Validation', 'Test'],
            linewidth=0.5,
            showfliers=False,  # exclude outliers
        )
        ax.set(
            xlabel=f'Nominal power, {glossary["power"]} [{units["power"]}]',
            ylabel='Brightness',
        )

        f.savefig(
            _image_brightness_study_path()
            / f'boiling1d-{"direct" if direct else "indirect"}-boxen.pdf'
        )
        f.savefig(
            _results_path() / f'brightness-distribution-{"direct" if direct else "indirect"}.pdf'
        )


def _cached_data_getter():
    @cache(
        JSONAllocator(_image_brightness_study_path() / 'boiling1d', suffix='.csv'),
        saver=_dataframe_to_csv,
        loader=pd.read_csv,
        exceptions=(OSError, AttributeError),
    )
    def _get_data(*, direct_visualization) -> pd.DataFrame:
        datasets = baseline_boiling_dataset(direct_visualization=direct_visualization)
        return _sorted_boiling_datasets(datasets)

    return _get_data


def _dataframe_to_csv(df: pd.DataFrame, path: PathLike) -> None:
    df.to_csv(resolve(path, parents=True), index=False)


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
                    image.mean(),
                    target['elapsed_time'],
                    class_name,
                )
                for class_name, ds in (
                    ('Training', ds_train),
                    ('Validation', ds_val),
                    ('Test', ds_test),
                )
                for image, target in ds.prefetch(1024)
                if DEFAULT_BOILING_OUTLIER_FILTER()(None, target)
            ),
            key=lambda power_hf_et_class: (
                power_hf_et_class[0],
                power_hf_et_class[2],
            ),
        ),
        columns=['nominal power', 'brightness', 'elapsed time', 'Subset'],
    )
    df['index'] = range(len(df))
    return df


def _image_brightness_study_path() -> Path:
    return resolve(studies_path() / 'image-brightness', dir=True)


def _results_path() -> Path:
    return resolve(figures_path() / 'results', dir=True)
