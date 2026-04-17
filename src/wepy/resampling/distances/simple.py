# Standard Library
import logging

# Third Party Library
import attrs
import numpy as np

# Local Modules
from .base import DistanceABC

logger = logging.getLogger(__name__)


@attrs.define
class XYDistanceState:
    coord: tuple[int, int]


@attrs.define
class XYEuclideanDistance(DistanceABC):
    """2 dimensional euclidean distance between points.

    States have the attributes 'x' and 'y'.

    """

    def image_distance(self, image_a, image_b):
        return np.sqrt(
            (image_a.coord[0] - image_b.coord[0]) ** 2
            + (image_a.coord[1] - image_b.coord[1]) ** 2
        )
