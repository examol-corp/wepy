from typing import Union, Literal

import numpy as np

FieldShapeSpec = Union[
    tuple[int, ...],
    Literal[Ellipsis],
    None,
]

FieldDtype = np.dtype | None
