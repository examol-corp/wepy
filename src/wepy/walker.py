"""Reference implementations for a general walker and walker state
with utilities for cloning and merging walkers.

Wepy does not require that this or a subclass of this `Walker` or
`WalkerState` is used and only that it is something that acts like it
(duck typed).

The required attributes for the Walker interface are:

- `state` : object implementing the `WalkerState` interface
- `weight` : float

The weight attribute is simply a float and the proper normalization of
weights among a weighted ensemble of walkers should be enforced by the
resampler.

The `clone`, `squash`, and `merge` methods can be accessed by Decision
classes for implementing cloning and merging. As can the module level
`split` and `keep_merge` functions.

The WalkerState interface must have a method `dict` which returns a
dictionary with string keys and arbitrary values.

Additionally the WalkerState should provide its own `__getitem__`
magic method for the accessor syntax, i.e. walker.state['positions'].

"""

# Standard Library
import logging
import math
import random as rand
from typing import Any, Generic, Protocol, TypeVar

# Third Party Library
import attrs

from wepy.missing import MISSING

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WalkerState(Protocol[T]):

    def __getitem__(self, key: str) -> T: ...

    def __eq__(self, other: Any) -> bool: ...

    def dict(self) -> dict[str, T]: ...

class AttrsWalkerStateMixin:
    """A convenient mixin for implementing the WalkerState interface
    for attrs classes."""

    def __getitem__(self, key: str) -> Any:
        if (value := getattr(self, key, MISSING)) is MISSING:
            raise KeyError(f"'key' '{key}' not found")
        else:
            return value

    def dict(self) -> dict[str, Any]:
        return attrs.asdict(self)
    

WalkerState_ = TypeVar("WalkerState_")


@attrs.define
class Walker(Generic[WalkerState_]):
    """Reference implementation of the Walker interface.

    A container for:

    - state
    - weight

    """

    state: WalkerState_
    weight: float = attrs.field(eq=attrs.cmp_using(eq=math.isclose))


def clone(walker: Walker, number: int = 1) -> list[Walker]:
    """Clone this walker by making a copy with the same state and split
    the probability uniformly between clones.

    The number is the increase in the number of walkers.

    e.g. number=1 will return 2 walkers with the same state as
    this object but with probability split 50/50 between them

    Parameters
    ----------
    number : int
        Number of extra clones to make
         (Default value = 1)

    Returns
    -------
    cloned_walkers : list of objects implementing the Walker interface

    """

    # calculate the weight of all child walkers split uniformly
    split_prob = walker.weight / (number + 1)
    # make the clones
    clones = []
    for i in range(number + 1):
        clones.append(Walker(walker.state, split_prob))

    return clones


def squash(walker: Walker, merge_target: Walker) -> Walker:
    """Add the weight of this walker to another.

    Parameters
    ----------
    merge_target : object implementing the Walker interface
        The walker to add this one's weight to.

    Returns
    -------
    merged_walker : object implementing the Walker interface

    """
    new_weight = walker.weight + merge_target.weight
    return Walker(merge_target.state, new_weight)


def split(walker: Walker, number: int = 2) -> list[Walker]:
    """Split (AKA make multiple clones) of a single walker.

    Creates multiple new walkers that have the same state as the given
    walker with weight evenly divided between them.

    Parameters
    ----------
    walker : object implementing the Walker interface
        The walker to split/clone
    number : int
        The number of clones to make of the walker
         (Default value = 2)

    Returns
    -------
    cloned_walkers : list of objects implementing the Walker interface

    """
    # calculate the weight of all child walkers split uniformly
    split_prob = walker.weight / (number)
    # make the clones
    clones = []
    for i in range(number):
        clones.append(type(walker)(walker.state, split_prob))

    return clones


def keep_merge(walkers: list[Walker], keep_idx: int) -> Walker:
    """Merge a set of walkers using the state of one of them.

    Parameters
    ----------
    walkers : list of objects implementing the Walker interface
        The walkers that will be merged together
    keep_idx : int
        The index of the walker in the walkers list that will be used
        to set the state of the new merged walker.

    Returns
    -------
    merged_walker : object implementing the Walker interface

    """

    weights = [walker.weight for walker in walkers]
    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(walkers[keep_idx].state, new_weight)

    return new_walker


def merge(walkers: list[Walker]) -> tuple[Walker, int]:
    """Merge this walker with another keeping the state of one of them
    and adding the weights.

    The walker that has it's state kept is a random choice weighted by
    the walkers weights.

    Parameters
    ----------
    walkers : The walkers that will be merged together

    Returns
    -------
    merged_walker : Final merged walker
    keep_idx: Index of the walker whose state was retained.

    """

    weights = [walker.weight for walker in walkers]
    # choose a walker according to their weights to keep its state
    keep_walker = next(iter(rand.choices(walkers, weights=weights)))
    keep_idx = walkers.index(keep_walker)

    # TODO do we need this?
    # the others are "squashed" and we lose their state
    # squashed_walkers = set(walkers).difference(keep_walker)

    # but we add their weight to the new walker
    new_weight = sum(weights)
    # create a new walker with the keep_walker state
    new_walker = type(walkers[0])(keep_walker.state, new_weight)

    return new_walker, keep_idx
