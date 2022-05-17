import abc
import math

from pint import Quantity


class Prism(abc.ABC):
    @abc.abstractmethod
    def length(self) -> Quantity[float]:
        pass

    @abc.abstractmethod
    def cross_section_perimeter(self) -> Quantity[float]:
        pass

    @abc.abstractmethod
    def cross_section_area(self) -> Quantity[float]:
        pass

    def lateral_area(self) -> Quantity[float]:
        return self.cross_section_perimeter() * self.length()

    def surface_area(self) -> Quantity[float]:
        return self.lateral_area() + 2 * self.cross_section_area()

    def volume(self) -> Quantity[float]:
        return self.cross_section_area() * self.length()


class Cylinder(Prism):
    def __init__(self, length: Quantity[float], diameter: Quantity[float]) -> None:
        self._length = length
        self._diameter = diameter

    def length(self) -> Quantity[float]:
        return self._length

    def diameter(self) -> Quantity[float]:
        return self._diameter

    def radius(self) -> Quantity[float]:
        return self._diameter / 2

    def cross_section_perimeter(self) -> Quantity[float]:
        return math.pi * self.diameter()

    def cross_section_area(self) -> Quantity[float]:
        return math.pi * self.radius() ** 2


class RectangularPrism(Prism):
    def __init__(
        self, length: Quantity[float], width: Quantity[float], thickness: Quantity[float]
    ) -> None:
        self._length = length
        self._width = width
        self._thickness = thickness

    def length(self) -> Quantity[float]:
        return self._length

    def width(self) -> Quantity[float]:
        return self._width

    def thickness(self) -> Quantity[float]:
        return self._thickness

    def cross_section_perimeter(self) -> Quantity[float]:
        return 2 * (self.width() + self.thickness())

    def cross_section_area(self) -> Quantity[float]:
        return self.width() * self.thickness()
