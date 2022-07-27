from boiling_learning.utils import geometry
from boiling_learning.utils.frozendicts import frozendict
from boiling_learning.utils.units import unit_registry as ureg

WIRE_SAMPLES = frozendict[int, geometry.Prism](
    {
        1: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.51 * ureg.millimeter),
        2: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.51 * ureg.millimeter),
        3: geometry.Cylinder(length=6.5 * ureg.centimeter, diameter=0.25 * ureg.millimeter),
        4: geometry.RectangularPrism(
            length=6.5 * ureg.centimeter,
            width=1 / 16 * ureg.inch,
            thickness=0.0031 * ureg.inch,
        ),
        5: geometry.RectangularPrism(
            length=6.5 * ureg.centimeter,
            width=1 / 16 * ureg.inch,
            thickness=0.0031 * ureg.inch,
        ),
    }
)
