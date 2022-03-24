from typing import List

from boiling_learning.preprocessing import ExperimentVideo, ImageDataset
from boiling_learning.utils import PathLike, print_header, resolve


def main(datapath: PathLike, verbose: bool = True) -> List[ImageDataset]:
    datapath = resolve(datapath)

    datasets: List[ImageDataset] = []
    for casedir in datapath.iterdir():
        if not casedir.is_dir():
            continue

        case = casedir.name
        for subcasedir in casedir.iterdir():
            if not subcasedir.is_dir():
                continue

            subcase = subcasedir.name

            dataset: ImageDataset = ImageDataset(f'{case}:{subcase}')
            for testdir in subcasedir.iterdir():
                test_name = testdir.name

                videopaths = (testdir / 'videos').glob('*.mp4')
                for video_path in videopaths:
                    video_name = video_path.stem
                    ev_name = ':'.join((case, subcase, test_name, video_name))
                    ev = ExperimentVideo(
                        df_path=video_path.with_suffix('.csv'),
                        video_path=video_path,
                        name=ev_name,
                    )
                    dataset.add(ev)

            if verbose:
                print_header(dataset.name)

            datasets.append(dataset)
    return datasets


if __name__ == '__main__':
    raise RuntimeError('*load_dataset_tree* cannot be executed as a standalone script yet.')
