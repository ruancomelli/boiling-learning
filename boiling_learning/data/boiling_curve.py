from typing import List, Tuple

from pint import Quantity

from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.units import unit_registry as ureg


@dataclass(frozen=True)
class NukiyamaBoilingCurve:
    """Experimental data published by Nukiyama.

    NUKIYAMA, Shiro. The maximum and minimum values of the heat Q transmitted from metal to boiling
    water under atmospheric pressure. International Journal of Heat and Mass Transfer, v. 9, p.
    1419-1433, Dec. 1966. Translated from Journal of the Society of Mechanical Engineers, 37,
    367-374, Japan, 1934. Available at http://www.htsj.or.jp/wp/media/IJHMT1984-3.pdf. Table 3.
    """

    material: str = 'Nichrome'
    diameter: Quantity[float] = 0.575 * ureg.mm
    length: Quantity[int] = 200 * ureg.mm

    @staticmethod
    def fetch() -> List[Tuple[Quantity[float], Quantity[float]]]:
        heat_flux = (
            [
                0.0527,
                1.385,
                5.44,
                12.66,
                22.42,
                27.15,
                32.43,
                35.29,
                38.22,
                40.48,
            ]
            * ureg.cal
            / (ureg.cm ** 2 * ureg.s)
        ).to(ureg.W / ureg.cm ** 2)

        temperature_excess = [
            3,
            8,
            13.5,
            18.8,
            25.7,
            31.0,
            35.5,
            38.0,
            44.0,
            46.5,
        ] * ureg.delta_degC

        return list(zip(heat_flux, temperature_excess))
