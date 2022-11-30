import numpy as np

from boiling_learning.preprocessing.image import grayscaler

if __name__ == '__main__':
    rgb_image = np.random.randint(0, 255, (3, 3, 3))

    grayscale = grayscaler()(rgb_image.astype(np.float32))

    print(rgb_image[..., 0])
    print(rgb_image[..., 1])
    print(rgb_image[..., 2])

    print()

    print(grayscale.squeeze())
