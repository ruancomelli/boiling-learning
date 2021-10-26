import math
from functools import partial
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from boiling_learning.preprocessing.image import downscale as tf_downscale
from boiling_learning.preprocessing.image import ensure_grayscale


def evaluate_downsampling(
    ref: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], float],
    downsamplers: Iterable[Callable[[np.ndarray], np.ndarray]],
) -> List[float]:
    pairs = ((ref, downsampler(ref)) for downsampler in downsamplers)
    evaluations = (evaluator(ref, image) for ref, image in pairs)

    return list(evaluations)


def downscale(image: np.ndarray, factor: int) -> np.ndarray:
    return tf_downscale(image, factors=(factor, factor)).numpy()


def main(
    image: np.ndarray,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    downscale_factors: Iterable[int] = (),
    final_downscale_factor: int = 5,
    xscale: str = 'log',
    figsize: Tuple[int, int] = (7, 5),
) -> None:
    image = ensure_grayscale(image)
    downscale_factors = sorted(frozenset(downscale_factors) | {1, final_downscale_factor})

    for name, scorer in metrics.items():
        ev_ds = evaluate_downsampling(
            image,
            scorer,
            [partial(downscale, factor=ds) for ds in downscale_factors],
        )

        evaluations = dict(zip(downscale_factors, ev_ds))
        original_evaluation = evaluations[1]
        final_evaluation = evaluations[final_downscale_factor]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(downscale_factors, ev_ds, color='k', s=15)
        ax.scatter(1, original_evaluation, color='r', s=25, label='reference')
        ax.scatter(
            final_downscale_factor,
            final_evaluation,
            color='b',
            s=25,
            label='target',
        )

        ax.legend()
        ax.set_xlabel('Downsampling factor')
        ax.set_title(
            f'{name} ({final_downscale_factor} -> {final_evaluation / original_evaluation:.0%})'
        )

        ax.set_xscale(xscale)

        _, top = ax.get_ylim()
        ax.set_ylim(0, math.ceil(top))

        ax.grid(which='both', axis='both', alpha=0.5)

        fig.show()

    # ------------------------------------------
    # PART 2
    # ------------------------------------------
    fig = plt.figure()
    fig.suptitle(f'Downscale factor: {final_downscale_factor}')

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title(str(np.squeeze(image).shape))

    ax = fig.add_subplot(1, 2, 2)
    downscaled = downscale(image, factor=final_downscale_factor)
    ax.imshow(downscaled, cmap='gray')
    ax.set_title(str(np.squeeze(downscaled).shape))


if __name__ == '__main__':
    raise RuntimeError('Cannot execute downscale analysis without an image!')
