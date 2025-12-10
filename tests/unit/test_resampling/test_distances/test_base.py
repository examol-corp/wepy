import math
import attrs

from wepy.resampling.distances.base import DistanceABC
from wepy.runners.mock import MockState

# minimal implementation of a Distance from the ABC, in this case the
# image and state are the same

@attrs.define
class MockDistance(DistanceABC):

    def image_distance(self, image_a: MockState, image_b: MockState) -> float:

        return math.sqrt((image_a.a - image_b.a) ** 2)

class Test_DistanceABC:

    def test_image(self):
        assert DistanceABC().image(MockState(1)) == MockState(1)
        assert MockDistance().image(MockState(1)) == MockState(1)

    def test_image_distance(self):
        assert math.isclose(
            MockDistance().image_distance(
                MockState(1),
                MockState(3),
            ),
            2.
        )

    def test_image_distance(self):
        assert math.isclose(
            MockDistance().distance(
                MockState(1),
                MockState(3),
            ),
            2.
        )
