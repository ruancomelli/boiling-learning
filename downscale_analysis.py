from functools import partial

import matplotlib.pyplot as plt
from more_itertools import first
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imread, imshow

from boiling_learning.preprocessing.image import crop, downscale


def evaluate_downsampling(img, evaluator, downsamplers):
    return [
        evaluator(img, downsampler(img))
        for downsampler in downsamplers
    ]

def img_variance(img):
    from numpy import var

    return var(img, axis=(0, 1))

def img_retained_variance(ref, img):
    return (
        img_variance(img)
        / img_variance(ref)
    )

def img_shannon_cross_entropy(ref, img):
    pass

def img_shannon_cross_entropy_ratio(ref, img):
    return (
        img_shannon_cross_entropy(ref, img)
        / img_shannon_cross_entropy(ref, ref)
    )

img_path = first((case.path / 'frames_crop').glob('**/*.png'))
print(img_path)
img = rgb2gray(imread(img_path))

ev_ds = evaluate_downsampling(
    img,
    img_retained_variance,
    [
        partial(downscale, shape=ds)
        for ds in range(1, 11)
    ]
)
print(ev_ds)

# ------------------------------------------
# PART 2
# ------------------------------------------

in_path = python_project_home_path / 'testing_extract' / 'from_0_to_9' / 'my_frame_1.png'

plt.subplot(1, 3, 1)

img = imread(in_path)
imshow(img)

top = 600
bottom = 230
left = 1000
right = 1100

plt.subplot(1, 3, 2)

cropped = crop(
    img,
    top=top,
    bottom=bottom,
    left=left,
    right=right,
)
print(cropped.shape)
imshow(cropped)

plt.subplot(1, 3, 3)

img = img_as_float(img)
downscaled = downscale(
    img,
    5
)
print(downscaled.shape)
imshow(downscaled)
