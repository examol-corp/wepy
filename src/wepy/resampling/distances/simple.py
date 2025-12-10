# Standard Library
import logging
from abc import ABC
from typing import TypeVar, Generic, Protocol

import numpy as np
import attrs

# First Party Library
from wepy.walker import WalkerState
from wepy.util.util import box_vectors_to_lengths_angles
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
        return np.sqrt((image_a.coord[0] - image_b.coord[0]) ** 2 + (image_a.coord[1] - image_b.coord[1]) ** 2)
