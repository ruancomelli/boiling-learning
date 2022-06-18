from typing import Dict, List, Union

from boiling_learning.preprocessing.image import (
    ConvertImageDType,
    Cropper,
    Downscaler,
    Grayscaler,
    RandomCropper,
)
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import VideoFrame

ExperimentVideoName = str


def main(
    downscale_factor: int = 5,
    height: int = 8 * 12,
    width: int = 8 * 12,
) -> List[
    Union[
        Transformer[VideoFrame, VideoFrame],
        Dict[ExperimentVideoName, Transformer[VideoFrame, VideoFrame]],
    ]
]:
    return [
        {
            'stainless steel:polished:test 6:00003': Cropper(
                left=849, right=1427, top=307, bottom=900
            ),
            'stainless steel:polished:test 6:00004': Cropper(
                left=854, right=1432, top=302, bottom=882
            ),
            'stainless steel:polished:test 7:00001': Cropper(
                left=689, right=1241, top=292, bottom=868
            ),
            'stainless steel:polished:test 7:00000': Cropper(
                left=684, right=1236, top=307, bottom=882
            ),
            'stainless steel:polished:test 4:00003': Cropper(
                left=789, right=1382, top=184, bottom=802
            ),
            'stainless steel:polished:test 4:00002': Cropper(
                left=784, right=1377, top=203, bottom=811
            ),
            'parametric:old:test 3:00006': Cropper(left=648, right=1206, top=307, bottom=872),
            'parametric:old:test 3:00007': Cropper(left=653, right=1201, top=307, bottom=868),
            'parametric:T_inf 40C:test 2:00004': Cropper(
                left=558, right=1116, top=288, bottom=849
            ),
            'parametric:T_inf 40C:test 2:00005': Cropper(
                left=553, right=1111, top=274, bottom=835
            ),
            'parametric:T_inf 40C:test 1:00000': Cropper(
                left=689, right=1252, top=354, bottom=929
            ),
            'parametric:T_inf 40C:test 1:00001': Cropper(
                left=689, right=1252, top=344, bottom=920
            ),
            'parametric:T_inf 40C:test 3:00006': Cropper(
                left=568, right=1116, top=297, bottom=839
            ),
            'parametric:T_inf 40C:test 3:00007': Cropper(
                left=568, right=1116, top=278, bottom=839
            ),
            'parametric:T_inf 60C:test 1:00000': Cropper(
                left=704, right=1257, top=264, bottom=821
            ),
            'parametric:T_inf 60C:test 1:00001': Cropper(
                left=704, right=1252, top=259, bottom=816
            ),
            'parametric:T_inf 60C:test 2:00003': Cropper(
                left=684, right=1252, top=250, bottom=811
            ),
            'parametric:T_inf 60C:test 2:00002': Cropper(
                left=704, right=1272, top=255, bottom=811
            ),
            'parametric:T_inf 60C:test 3:00006': Cropper(
                left=724, right=1257, top=297, bottom=839
            ),
            'parametric:T_inf 60C:test 3:00007': Cropper(
                left=729, right=1262, top=297, bottom=830
            ),
            'parametric:T_s 5C:test 3:00007': Cropper(left=638, right=1201, top=222, bottom=783),
            'parametric:T_s 5C:test 3:00008': Cropper(left=628, right=1186, top=217, bottom=778),
            'parametric:T_s 5C:test 2:00006': Cropper(left=467, right=1025, top=217, bottom=792),
            'parametric:T_s 5C:test 2:00005': Cropper(left=467, right=1035, top=222, bottom=802),
            'parametric:T_s 5C:test 1:00000': Cropper(left=849, right=1417, top=226, bottom=783),
            'parametric:T_s 5C:test 1:00001': Cropper(left=859, right=1417, top=212, bottom=778),
            'parametric:T_s 20C:test 3:00007': Cropper(left=749, right=1302, top=274, bottom=830),
            'parametric:T_s 20C:test 3:00008': Cropper(left=759, right=1312, top=264, bottom=821),
            'parametric:T_s 20C:test 1:00000': Cropper(left=638, right=1206, top=292, bottom=872),
            'parametric:T_s 20C:test 1:00001': Cropper(left=633, right=1211, top=274, bottom=858),
            'parametric:T_s 20C:test 2:00006': Cropper(left=875, right=1407, top=203, bottom=750),
            'parametric:T_s 20C:test 2:00005': Cropper(left=870, right=1412, top=222, bottom=745),
            'parametric:rh 70%:test 3:00000': Cropper(left=704, right=1277, top=307, bottom=868),
            'parametric:rh 70%:test 3:00001': Cropper(left=694, right=1267, top=297, bottom=868),
            'parametric:rh 70%:test 7:00002': Cropper(left=684, right=1246, top=179, bottom=726),
            'parametric:rh 70%:test 7:00003': Cropper(left=684, right=1246, top=184, bottom=722),
            'parametric:rh 70%:test 6:00000': Cropper(left=668, right=1226, top=222, bottom=764),
            'parametric:rh 70%:test 6:00001': Cropper(left=658, right=1216, top=208, bottom=755),
            'parametric:rh 90%:test 3:00009': Cropper(left=628, right=1181, top=311, bottom=877),
            'parametric:rh 90%:test 3:00008': Cropper(left=623, right=1176, top=316, bottom=877),
            'parametric:rh 90%:test 2:00007': Cropper(left=638, right=1206, top=373, bottom=953),
            'parametric:rh 90%:test 2:00006': Cropper(left=648, right=1196, top=373, bottom=948),
            'parametric:rh 90%:test 5:00013': Cropper(left=744, right=1307, top=231, bottom=802),
            'parametric:rh 90%:test 5:00012': Cropper(left=744, right=1312, top=226, bottom=806),
        },
        ConvertImageDType('float32'),
        Grayscaler(),
        Downscaler(downscale_factor),
        RandomCropper(height=height, width=width),
    ]


if __name__ == '__main__':
    raise RuntimeError(
        '*make_condensation_processors* cannot be executed as a standalone script yet.'
    )
