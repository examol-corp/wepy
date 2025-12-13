"""Defines a generic protocol for storage of data.

This should only define the interfaces between the generating
components (resamplers, runner, boundary conditions, sim_manager) and
the reporting and storage backends should utilize.

"""
from collections.abc import Mapping
from numpy.typing import NDArray
from typing import Union, Literal

import numpy as np

RecordValueDtype = int | float | NDArray
Record = Mapping[str, RecordValueDtype]

# Numpy-style shapes of all fields produced in records.
#
# There should be the same number of elements as there are in the
# corresponding 'FIELDS' class constant.
#
# Each entry should either be:
#
# A. A tuple of ints that specify the shape of the field element
#    array.
#
# B. Ellipsis, indicating that the field is variable length and
#    limited to being a rank one array (e.g. (3,) or (1,)).
# Note that the shapes must be tuple and not simple integers for rank-1
# arrays.
#
# Option B will result in the special h5py datatype 'vlen' and
# should not be used for large datasets for efficiency reasons.
RecordFieldShapeSpec = Union[
    tuple[int, ...],
    Literal[Ellipsis],
]



# There should be the same number of elements as there are in the
# corresponding 'FIELDS' class constant.
#
# Each entry should either be:
#
# A. A `numpy.dtype` object.

RecordFieldDtype = Union[
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.bool,
] 


RecordFieldSpec = tuple[
    str, # name
    RecordFieldShapeSpec, # shape
    RecordFieldDtype, # dtype
]
