from dataclasses import dataclass, field

import numpy as np
import pint


@dataclass(frozen=True)
class Solid:
    surface_area: pint.Quantity = field(init=False)
    volume: pint.Quantity = field(init=False)


@dataclass(frozen=True)
class Prism(Solid):
    length: pint.Quantity

    cross_section_perimeter: pint.Quantity = field(init=False)
    cross_section_area: pint.Quantity = field(init=False)

    lateral_area: pint.Quantity = field(init=False)
    surface_area: pint.Quantity = field(init=False)
    volume: pint.Quantity = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'lateral_area', self.cross_section_perimeter * self.length)
        object.__setattr__(self, 'surface_area', self.lateral_area + 2*self.cross_section_area)
        object.__setattr__(self, 'volume', self.cross_section_area * self.length)


@dataclass(frozen=True)
class Cylinder(Prism):
    diameter: pint.Quantity

    def __post_init__(self):
        object.__setattr__(self, 'radius', self.diameter / 2)
        object.__setattr__(self, 'cross_section_perimeter', np.pi * self.diameter)
        object.__setattr__(self, 'cross_section_area', np.pi * self.radius**2)
        super().__post_init__()


@dataclass(frozen=True)
class RectangularPrism(Prism):
    width: pint.Quantity
    thickness: pint.Quantity

    def __post_init__(self):
        object.__setattr__(self, 'cross_section_perimeter', 2*(self.width + self.thickness))
        object.__setattr__(self, 'cross_section_area', self.width * self.thickness)
        super().__post_init__()
