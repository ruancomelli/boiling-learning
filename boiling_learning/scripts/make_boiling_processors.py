from typing import List, Tuple

import tensorflow as tf

from boiling_learning.preprocessing.image import (
    crop,
    downscale,
    random_brightness,
    random_crop,
    shrink,
)
from boiling_learning.preprocessing.transformers import (
    DictImageTransformer,
    ImageTransformer,
    Transformer,
)
from boiling_learning.utils.functional import P


def make_boiling_processors(
    direct_visualization: bool = True,
    downscale_factor: int = 5,
    direct_height: int = 180,
    indirect_height: int = 108,
    indirect_height_ratio: float = 0.4,
    width: int = 128,
) -> Tuple[List[Transformer], List[Transformer]]:
    preprocessors = [
        ImageTransformer('grayscaler', tf.image.rgb_to_grayscale),
        DictImageTransformer(
            'region_cropper',
            crop,
            {
                'GOPR2819': P(left=861, right=1687, top=321, bottom=1273),
                'GOPR2820': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2821': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2822': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2823': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2824': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2825': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2826': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2827': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2828': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2829': P(left=861, right=1678, top=321, bottom=1267),
                'GOPR2830': P(left=880, right=1678, top=350, bottom=1267),
                'GOPR2831': P(left=880, right=1678, top=350, bottom=1267),
                'GOPR2832': P(left=880, right=1678, top=350, bottom=1267),
                'GOPR2833': P(left=880, right=1678, top=350, bottom=1267),
                'GOPR2834': P(left=880, right=1678, top=350, bottom=1250),
                'GOPR2835': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2836': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2837': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2838': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2839': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2840': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2841': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2842': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2843': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2844': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2845': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2846': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2847': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2848': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2849': P(left=880, right=1678, top=350, bottom=1260),
                'GOPR2852': P(left=1010, right=1865, top=150, bottom=1240),
                'GOPR2853': P(left=1010, right=1865, top=150, bottom=1240),
                'GOPR2854': P(left=1010, right=1865, top=150, bottom=1240),
                'GOPR2855': P(left=1010, right=1865, top=150, bottom=1240),
                'GOPR2856': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2857': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2858': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2859': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2860': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2861': P(left=1040, right=1890, top=150, bottom=1240),
                'GOPR2862': P(left=1040, right=1890, top=250, bottom=1240),
                'GOPR2863': P(left=1040, right=1890, top=250, bottom=1240),
                'GOPR2864': P(left=1040, right=1890, top=250, bottom=1240),
                'GOPR2865': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2866': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2867': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2868': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2869': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2870': P(left=1040, right=1890, top=340, bottom=1240),
                'GOPR2873': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2874': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2875': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2876': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2877': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2878': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2879': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2880': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2881': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2882': P(left=1000, right=1700, top=350, bottom=1250),
                'GOPR2884': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2885': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2886': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2887': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2888': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2889': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2890': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2891': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2892': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2893': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2894': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2895': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2896': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2897': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2898': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2899': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2900': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2901': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2902': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2903': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2904': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2905': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2906': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2907': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2908': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2909': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2910': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2911': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2914': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2915': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2916': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2917': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2918': P(left=960, right=1750, top=400, bottom=1350),
                'GOPR2919': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2920': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2921': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2922': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2923': P(left=960, right=1750, top=450, bottom=1350),
                'GOPR2925': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2926': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2927': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2928': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2929': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2930': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2931': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2932': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2933': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2934': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2935': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2936': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2937': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2938': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2939': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2940': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2941': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2942': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2943': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2944': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2945': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2946': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2947': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2948': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2949': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2950': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2951': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2952': P(left=980, right=1810, top=300, bottom=1350),
                'GOPR2953': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2954': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2955': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2956': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2957': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2959': P(left=980, right=1810, top=400, bottom=1350),
                'GOPR2960': P(left=980, right=1810, top=400, bottom=1350),
            },
        ),
        ImageTransformer(
            'downscaler',
            downscale,
            pack=P((downscale_factor, downscale_factor, 1), antialias=True),
        ),
        ImageTransformer(
            'visualization_shrinker',
            shrink,
            pack=P(
                left=0,
                right=0,
                top=0,
                bottom=(0 if direct_visualization else indirect_height_ratio),
            ),
        ),
        ImageTransformer(
            'final_height_shrinker',
            shrink,
            pack=P(
                left=0,
                right=0,
                bottom=0,
                height=(
                    direct_height if direct_visualization else indirect_height
                ),
            ),
        ),
    ]

    augmentors = [
        ImageTransformer(
            'random_cropper', random_crop, pack=P((None, width, None))
        ),
        ImageTransformer(
            'random_left_right_flipper', tf.image.flip_left_right
        ),
        ImageTransformer(
            'random_brightness', random_brightness, pack=P(-0.2, 0.2)
        ),
        ImageTransformer(
            'random_contrast', tf.image.random_contrast, pack=P(0.6, 1.4)
        ),
        ImageTransformer(
            'random_quality', tf.image.random_jpeg_quality, pack=P(30, 100)
        ),
    ]

    return preprocessors, augmentors


if __name__ == '__main__':
    raise RuntimeError(
        '*make_boiling_processors* cannot be executed as a standalone script yet.'
    )
