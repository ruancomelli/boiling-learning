from functools import partial
from typing import Callable, Dict, Iterable, List

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
) -> None:
    image = ensure_grayscale(image)
    downscale_factors = sorted(
        frozenset(downscale_factors) | {1, final_downscale_factor}
    )

    for name, scorer in metrics.items():
        ev_ds = evaluate_downsampling(
            image,
            scorer,
            [partial(downscale, factor=ds) for ds in downscale_factors],
        )

        evaluations = dict(zip(downscale_factors, ev_ds))
        original_evaluation = evaluations[1]
        final_evaluation = evaluations[final_downscale_factor]

        fig, ax = plt.subplots()
        ax.plot(downscale_factors, ev_ds)
        ax.axvline(final_downscale_factor, linestyle='--', color='k')
        ax.set_title(
            f'{name} ({final_downscale_factor} -> {final_evaluation / original_evaluation:.0%})'
        )
        ax.set_xscale('log')

        fig.show()

    # ------------------------------------------
    # PART 2
    # ------------------------------------------
    fig = plt.figure()
    fig.suptitle(f'Downscale factor: {final_downscale_factor}')

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title(str(np.squeeze(image).shape))

    ax = fig.add_subplot(1, 3, 2)
    downscaled = downscale(image, factor=final_downscale_factor)
    ax.imshow(downscaled, cmap='gray')
    ax.set_title(str(np.squeeze(downscaled).shape))


if __name__ == '__main__':
    raise RuntimeError('Cannot execute downscale analysis without an image!')
