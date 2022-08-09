from fractions import Fraction
from typing import Dict, List, Optional, Union

from typing_extensions import Literal

from boiling_learning.preprocessing.image import (
    CenterCropper,
    ConvertImageDType,
    Cropper,
    Downscaler,
    Grayscaler,
    RandomCropper,
    VideoFrameOrFrames,
)
from boiling_learning.preprocessing.transformers import Operator

ExperimentVideoName = str

# equal to Hobold's:
RECOMMENDED_DIRECT_HEIGHT = 120
RECOMMENDED_INDIRECT_HEIGHT = 72
RECOMMENDED_WIDTH = 196


def main(
    direct_visualization: bool = True,
    downscale_factor: int = 4,
    height: Optional[int] = None,
    width: int = RECOMMENDED_WIDTH,
    visualization_window_width: Fraction = Fraction(1, 1),
    crop_mode: Literal['center', 'random'] = 'center',
) -> List[
    Union[
        Operator[VideoFrameOrFrames],
        Dict[ExperimentVideoName, Operator[VideoFrameOrFrames]],
    ]
]:
    if height is None:
        height = RECOMMENDED_DIRECT_HEIGHT if direct_visualization else RECOMMENDED_INDIRECT_HEIGHT

    return [
        ConvertImageDType('float32'),
        Grayscaler(),
        {
            'GOPR2819': Cropper(left=861, right=1687, top=321, bottom=1190),
            'GOPR2820': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2821': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2822': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2823': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2824': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2825': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2826': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2827': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2828': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2829': Cropper(left=861, right=1678, top=321, bottom=1180),
            'GOPR2830': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2831': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2832': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2833': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2834': Cropper(left=880, right=1678, top=350, bottom=1170),
            'GOPR2835': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2836': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2837': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2838': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2839': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2840': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2841': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2842': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2843': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2844': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2845': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2846': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2847': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2848': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2849': Cropper(left=880, right=1678, top=350, bottom=1180),
            'GOPR2852': Cropper(left=1010, right=1865, top=150, bottom=1160),
            'GOPR2853': Cropper(left=1010, right=1865, top=150, bottom=1160),
            'GOPR2854': Cropper(left=1020, right=1865, top=150, bottom=1160),
            'GOPR2855': Cropper(left=1020, right=1865, top=150, bottom=1160),
            'GOPR2856': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2857': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2858': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2859': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2860': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2861': Cropper(left=1040, right=1890, top=150, bottom=1160),
            'GOPR2862': Cropper(left=1040, right=1890, top=250, bottom=1160),
            'GOPR2863': Cropper(left=1040, right=1890, top=250, bottom=1160),
            'GOPR2864': Cropper(left=1040, right=1890, top=250, bottom=1160),
            'GOPR2865': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2866': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2867': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2868': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2869': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2870': Cropper(left=1040, right=1890, top=340, bottom=1160),
            'GOPR2873': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2874': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2875': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2876': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2877': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2878': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2879': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2880': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2881': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2882': Cropper(left=990, right=1780, top=350, bottom=1220),
            'GOPR2884': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2885': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2886': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2887': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2888': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2889': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2890': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2891': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2892': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2893': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2894': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2895': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2896': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2897': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2898': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2899': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2900': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2901': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2902': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2903': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2904': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2905': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2906': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2907': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2908': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2909': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2910': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2911': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2914': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2915': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2916': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2917': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2918': Cropper(left=960, right=1750, top=400, bottom=1250),
            'GOPR2919': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2920': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2921': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2922': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2923': Cropper(left=960, right=1750, top=450, bottom=1250),
            'GOPR2925': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2926': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2927': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2928': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2929': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2930': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2931': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2932': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2933': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2934': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2935': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2936': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2937': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2938': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2939': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2940': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2941': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2942': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2943': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2944': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2945': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2946': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2947': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2948': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2949': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2950': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2951': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2952': Cropper(left=980, right=1810, top=300, bottom=1250),
            'GOPR2953': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2954': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2955': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2956': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2957': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2959': Cropper(left=980, right=1810, top=400, bottom=1250),
            'GOPR2960': Cropper(left=980, right=1810, top=400, bottom=1250),
        },
        Cropper(
            height=height * downscale_factor,
            bottom_border=0,
            left=0,
            right_border=0,
        ),
        # cropping width is done deterministically here for validating with the literature
        # however, ideally we should be able to choose if we want a deterministic or a randomic
        # crop
        {'center': CenterCropper, 'random': RandomCropper}[crop_mode](
            width=width * downscale_factor
        ),
        Downscaler(downscale_factor),
        {'center': CenterCropper, 'random': RandomCropper}[crop_mode](
            width=round(width * visualization_window_width)
        ),
    ]


if __name__ == '__main__':
    raise RuntimeError('*make_boiling_processors* cannot be executed as a standalone script yet.')
