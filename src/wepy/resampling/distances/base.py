"""Modular component for defining distance metrics usable within many
different resamplers.

This module contains an abstract base class for Distance classes.

The suggested implementation method is to leave the 'distance' method
as is, and override the 'image' and 'image_distance' methods
instead. Because the default 'distance' method calls these
transparently. The resamplers will use the 'image' and
'image_distance' calls because this allows performance optimizations.

For example in WExplore the images for some walkers end up being
stored as the definitions of Voronoi regions, and if the whole walker
state was stored it would not only use much more space in memory but
require that common transformations be repeated every time a distance
is to be calculated to that image (which is many times). In REVO all
to all distances between walkers are computed which would also incur a
high cost.

So use the 'image' to do precomputations on raw walker states and use
the 'image_distance' to compute distances using only those images.

"""

# Standard Library
import logging
from abc import ABC
from typing import TypeVar, Generic, Protocol

# First Party Library
from wepy.walker import WalkerState
from wepy.util.util import box_vectors_to_lengths_angles

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState, covariant=True)
DistanceImage_ = TypeVar("DistanceImage_")

class Distance(Protocol[DistanceImage_, WalkerState_]):
    def image(self, state: WalkerState_) -> DistanceImage_:
        """Compute the 'image' of a walker state which should be some
        transformation of the walker state that is more
        convenient. E.g. for precomputation of expensive operations or
        for saving as resampler state.

        The abstract implementation is naive and just returns the
        state itself, thus it is the identity function.

        Parameters
        ----------
        state : object implementing WalkerState
            The state which will be transformed to an image

        Returns
        -------
        image : object implementing WalkerState
            The same state that was given as an argument.

        """

        ...

    def image_distance(self, image_a: DistanceImage_, image_b: DistanceImage_) -> float:
        """Compute the distance between two images of walker states.

        Parameters
        ----------
        image_a : object produced by Distance.image

        image_b : object produced by Distance.image

        Returns
        -------
        distance : float
            The distance between the two images

        Raises
        ------
        NotImplementedError : always because this is abstract

        """
        ...

    def distance(self, state_a: WalkerState_, state_b: WalkerState_) -> float:
        """Compute the distance between two states.

        Parameters
        ----------
        state_a : object implementing WalkerState

        state_b : object implementing WalkerState

        Returns
        -------
        distance : float
            The distance between the two walker states


        """

        ...


        

class DistanceABC(ABC, Generic[DistanceImage_, WalkerState_]):
    """Abstract Base class for Distance classes."""

    def image(self, state: WalkerState_) -> DistanceImage_:
        return state

    def image_distance(self, image_a: DistanceImage_, image_b: DistanceImage_) -> float:
        raise NotImplementedError

    def distance(self, state_a: WalkerState_, state_b: WalkerState_) -> float:
        return self.image_distance(self.image(state_a), self.image(state_b))
