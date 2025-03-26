from fractions import Fraction
from typing import Final, Literal, TypeAlias

from boiling_learning.preprocessing.image import (
    VideoFrameOrFrames,
    center_cropper,
    cropper,
    downscaler,
    grayscaler,
    image_dtype_converter,
    random_cropper,
)
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils.mathutils import round_to_multiple

ExperimentVideoName: TypeAlias = str

# equal to Hobold's:
DEFAULT_DIRECT_HEIGHT: Final = 120
DEFAULT_INDIRECT_HEIGHT: Final = 72
DEFAULT_WIDTH: Final = 196
DEFAULT_DOWNSCALE_FACTOR: Final = 5
DEFAULT_VISUALIZATION_WINDOW_WIDTH: Final = Fraction(60, 100)

# equal to Hobold's:
RECOMMENDED_DIRECT_HEIGHT: Final = 120
RECOMMENDED_INDIRECT_HEIGHT: Final = 72
RECOMMENDED_WIDTH: Final = 196
RECOMMENDED_DOWNSCALE_FACTOR: Final = 5
RECOMMENDED_VISUALIZATION_WINDOW_WIDTH: Final = Fraction(60, 100)


def default_boiling_preprocessors(
    direct_visualization: bool = True,
    downscale_factor: int = DEFAULT_DOWNSCALE_FACTOR,
    height: int | None = None,
    bottom_border: int | None = None,
    width: int = DEFAULT_WIDTH,
    visualization_window_width: Fraction = DEFAULT_VISUALIZATION_WINDOW_WIDTH,
    crop_mode: Literal["center", "random"] = "center",
) -> list[
    list[
        Transformer[VideoFrameOrFrames, VideoFrameOrFrames]
        | dict[ExperimentVideoName, Transformer[VideoFrameOrFrames, VideoFrameOrFrames]]
    ]
]:
    if height is None:
        height = (
            DEFAULT_DIRECT_HEIGHT if direct_visualization else DEFAULT_INDIRECT_HEIGHT
        )

    if bottom_border is None:
        bottom_border = DEFAULT_DIRECT_HEIGHT - height

    return [
        [
            image_dtype_converter("float32"),
            {
                "GOPR2819": cropper(left=861, right=1687, top=321, bottom=1150),
                "GOPR2820": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2821": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2822": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2823": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2824": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2825": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2826": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2827": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2828": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2829": cropper(left=861, right=1678, top=321, bottom=1140),
                "GOPR2830": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2831": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2832": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2833": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2834": cropper(left=880, right=1678, top=350, bottom=1130),
                "GOPR2835": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2836": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2837": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2838": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2839": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2840": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2841": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2842": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2843": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2844": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2845": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2846": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2847": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2848": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2849": cropper(left=880, right=1678, top=350, bottom=1140),
                "GOPR2852": cropper(left=1010, right=1865, top=150, bottom=1120),
                "GOPR2853": cropper(left=1010, right=1865, top=150, bottom=1120),
                "GOPR2854": cropper(left=1020, right=1865, top=150, bottom=1120),
                "GOPR2855": cropper(left=1020, right=1865, top=150, bottom=1120),
                "GOPR2856": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2857": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2858": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2859": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2860": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2861": cropper(left=1040, right=1890, top=150, bottom=1120),
                "GOPR2862": cropper(left=1040, right=1890, top=250, bottom=1120),
                "GOPR2863": cropper(left=1040, right=1890, top=250, bottom=1120),
                "GOPR2864": cropper(left=1040, right=1890, top=250, bottom=1120),
                "GOPR2865": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2866": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2867": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2868": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2869": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2870": cropper(left=1040, right=1890, top=340, bottom=1120),
                "GOPR2873": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2874": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2875": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2876": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2877": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2878": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2879": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2880": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2881": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2882": cropper(left=990, right=1780, top=350, bottom=1220),
                "GOPR2884": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2885": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2886": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2887": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2888": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2889": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2890": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2891": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2892": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2893": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2894": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2895": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2896": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2897": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2898": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2899": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2900": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2901": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2902": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2903": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2904": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2905": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2906": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2907": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2908": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2909": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2910": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2911": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2914": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2915": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2916": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2917": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2918": cropper(left=960, right=1750, top=400, bottom=1250),
                "GOPR2919": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2920": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2921": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2922": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2923": cropper(left=960, right=1750, top=450, bottom=1250),
                "GOPR2925": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2926": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2927": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2928": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2929": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2930": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2931": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2932": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2933": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2934": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2935": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2936": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2937": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2938": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2939": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2940": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2941": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2942": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2943": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2944": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2945": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2946": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2947": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2948": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2949": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2950": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2951": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2952": cropper(left=980, right=1810, top=300, bottom=1200),
                "GOPR2953": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2954": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2955": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2956": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2957": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2959": cropper(left=980, right=1810, top=400, bottom=1200),
                "GOPR2960": cropper(left=980, right=1810, top=400, bottom=1200),
            },
            grayscaler(),
        ],
        [
            downscaler(downscale_factor),
            # round for extra speed -- NVIDIA's Tensor Cores benefit a lot from this
            cropper(
                height=round_to_multiple(height, base=8), bottom_border=bottom_border
            ),
            {"center": center_cropper, "random": random_cropper}[crop_mode](
                width=round_to_multiple(width * visualization_window_width, base=8),
            ),
        ],
    ]


def default_condensation_preprocessors(
    downscale_factor: int = 5,
    height: int = 8 * 12,
    width: int = 8 * 12,
    crop_mode: Literal["center", "random"] = "center",
) -> list[
    Transformer[VideoFrameOrFrames, VideoFrameOrFrames]
    | dict[ExperimentVideoName, Transformer[VideoFrameOrFrames, VideoFrameOrFrames]],
]:
    return [
        image_dtype_converter("float32"),
        {
            "stainless steel:polished:test 6:00003": cropper(
                left=849, right=1427, top=307, bottom=900
            ),
            "stainless steel:polished:test 6:00004": cropper(
                left=854, right=1432, top=302, bottom=882
            ),
            "stainless steel:polished:test 7:00001": cropper(
                left=689, right=1241, top=292, bottom=868
            ),
            "stainless steel:polished:test 7:00000": cropper(
                left=684, right=1236, top=307, bottom=882
            ),
            "stainless steel:polished:test 4:00003": cropper(
                left=789, right=1382, top=184, bottom=802
            ),
            "stainless steel:polished:test 4:00002": cropper(
                left=784, right=1377, top=203, bottom=811
            ),
            "parametric:old:test 3:00006": cropper(
                left=648, right=1206, top=307, bottom=872
            ),
            "parametric:old:test 3:00007": cropper(
                left=653, right=1201, top=307, bottom=868
            ),
            "parametric:T_inf 40C:test 2:00004": cropper(
                left=558, right=1116, top=288, bottom=849
            ),
            "parametric:T_inf 40C:test 2:00005": cropper(
                left=553, right=1111, top=274, bottom=835
            ),
            "parametric:T_inf 40C:test 1:00000": cropper(
                left=689, right=1252, top=354, bottom=929
            ),
            "parametric:T_inf 40C:test 1:00001": cropper(
                left=689, right=1252, top=344, bottom=920
            ),
            "parametric:T_inf 40C:test 3:00006": cropper(
                left=568, right=1116, top=297, bottom=839
            ),
            "parametric:T_inf 40C:test 3:00007": cropper(
                left=568, right=1116, top=278, bottom=839
            ),
            "parametric:T_inf 60C:test 1:00000": cropper(
                left=704, right=1257, top=264, bottom=821
            ),
            "parametric:T_inf 60C:test 1:00001": cropper(
                left=704, right=1252, top=259, bottom=816
            ),
            "parametric:T_inf 60C:test 2:00003": cropper(
                left=684, right=1252, top=250, bottom=811
            ),
            "parametric:T_inf 60C:test 2:00002": cropper(
                left=704, right=1272, top=255, bottom=811
            ),
            "parametric:T_inf 60C:test 3:00006": cropper(
                left=724, right=1257, top=297, bottom=839
            ),
            "parametric:T_inf 60C:test 3:00007": cropper(
                left=729, right=1262, top=297, bottom=830
            ),
            "parametric:T_s 5C:test 3:00007": cropper(
                left=638, right=1201, top=222, bottom=783
            ),
            "parametric:T_s 5C:test 3:00008": cropper(
                left=628, right=1186, top=217, bottom=778
            ),
            "parametric:T_s 5C:test 2:00006": cropper(
                left=467, right=1025, top=217, bottom=792
            ),
            "parametric:T_s 5C:test 2:00005": cropper(
                left=467, right=1035, top=222, bottom=802
            ),
            "parametric:T_s 5C:test 1:00000": cropper(
                left=849, right=1417, top=226, bottom=783
            ),
            "parametric:T_s 5C:test 1:00001": cropper(
                left=859, right=1417, top=212, bottom=778
            ),
            "parametric:T_s 20C:test 3:00007": cropper(
                left=749, right=1302, top=274, bottom=830
            ),
            "parametric:T_s 20C:test 3:00008": cropper(
                left=759, right=1312, top=264, bottom=821
            ),
            "parametric:T_s 20C:test 1:00000": cropper(
                left=638, right=1206, top=292, bottom=872
            ),
            "parametric:T_s 20C:test 1:00001": cropper(
                left=633, right=1211, top=274, bottom=858
            ),
            "parametric:T_s 20C:test 2:00006": cropper(
                left=875, right=1407, top=203, bottom=750
            ),
            "parametric:T_s 20C:test 2:00005": cropper(
                left=870, right=1412, top=222, bottom=745
            ),
            "parametric:rh 70%:test 3:00000": cropper(
                left=704, right=1277, top=307, bottom=868
            ),
            "parametric:rh 70%:test 3:00001": cropper(
                left=694, right=1267, top=297, bottom=868
            ),
            "parametric:rh 70%:test 7:00002": cropper(
                left=684, right=1246, top=179, bottom=726
            ),
            "parametric:rh 70%:test 7:00003": cropper(
                left=684, right=1246, top=184, bottom=722
            ),
            "parametric:rh 70%:test 6:00000": cropper(
                left=668, right=1226, top=222, bottom=764
            ),
            "parametric:rh 70%:test 6:00001": cropper(
                left=658, right=1216, top=208, bottom=755
            ),
            "parametric:rh 90%:test 3:00009": cropper(
                left=628, right=1181, top=311, bottom=877
            ),
            "parametric:rh 90%:test 3:00008": cropper(
                left=623, right=1176, top=316, bottom=877
            ),
            "parametric:rh 90%:test 2:00007": cropper(
                left=638, right=1206, top=373, bottom=953
            ),
            "parametric:rh 90%:test 2:00006": cropper(
                left=648, right=1196, top=373, bottom=948
            ),
            "parametric:rh 90%:test 5:00013": cropper(
                left=744, right=1307, top=231, bottom=802
            ),
            "parametric:rh 90%:test 5:00012": cropper(
                left=744, right=1312, top=226, bottom=806
            ),
        },
        grayscaler(),
        downscaler(downscale_factor),
        {"center": center_cropper, "random": random_cropper}[crop_mode](
            height=round_to_multiple(height, base=8),
            width=round_to_multiple(width, base=8),
        ),
    ]
