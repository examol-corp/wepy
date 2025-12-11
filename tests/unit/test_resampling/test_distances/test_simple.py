# Standard Library
import math

# First Party Library
from wepy.resampling.distances.simple import (
    XYDistanceState,
    XYEuclideanDistance,
)


class Test_XYEuclideanDistance:

    def test_image_distance(self):

        assert math.isclose(
            XYEuclideanDistance().image_distance(
                XYDistanceState((0, 0)),
                XYDistanceState((0, 2)),
            ),
            2.0,
        )

    def test_distance(self):

        assert math.isclose(
            XYEuclideanDistance().distance(
                XYDistanceState((0, 0)),
                XYDistanceState((0, 2)),
            ),
            2.0,
        )
