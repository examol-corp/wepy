"""Some type helpers.

Provides some Numpy array specifiers useful in Annotated that won't
actually be checked in a checker.

"""

# Standard Library
from typing import Annotated, Literal, Union

# Third Party Library
import attrs
import numpy as np
from numpy.typing import NDArray


@attrs.define
class Shape:
    dims: tuple[int | Literal[Ellipsis], ...]


IdxArray = Annotated[
    NDArray[np.integer],
    Shape((...,)),
]
Idxs = Union[
    list[int],
    IdxArray,
]
