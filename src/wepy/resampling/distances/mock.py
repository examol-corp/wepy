# Standard Library
import math

# Third Party Library
import attrs

# First Party Library
from wepy.runners.mock import MockState

# Local Modules
from .base import DistanceABC


@attrs.define
class MockDistance(DistanceABC):

    def image_distance(self, image_a: MockState, image_b: MockState) -> float:

        return math.sqrt((image_a.a - image_b.a) ** 2)
