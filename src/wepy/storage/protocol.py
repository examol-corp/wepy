"""Defines a generic protocol for storage of data.

This should only define the interfaces between the generating
components (resamplers, runner, boundary conditions, sim_manager) and
the reporting and storage backends should utilize.

"""
from collections.abc import Mapping
from numpy.typing import NDArray
from typing import Union, Literal, TypedDict, Required

import numpy as np
import attrs

RecordValueDtype = int | float | NDArray

Record = Mapping[str, RecordValueDtype]
@attrs.define
class RunRecord:
    cycle_idx: Required[int]
    record: Record

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
RecordFieldShape = tuple[int, ...]
RecordFieldShapeSpec = Union[
    RecordFieldShape,
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
    bool,
] 


RecordFieldSpec = tuple[
    str, # name
    RecordFieldShapeSpec, # shape
    RecordFieldDtype, # dtype
]


# Specific record types guaranteed

DECISION_RECORD_FIELDS = frozenset({
    "decision_id",
    "target_idxs",
})

class DecisionRecordUnstruct(TypedDict, total=False):
    decision_id: Required[int]
    target_idxs: Required[tuple[int, ...]]
    

RESAMPLING_RECORD_FIELDS = frozenset({
    "step_idx",
    "walker_idx",
    "decision_id",
    "target_idxs",
})

@attrs.define
class ResamplingRecord:
    step_idx: int
    walker_idx: int
    decision_id: int
    target_idxs: tuple[int, ...]

class ResamplingRecordUnstruct(TypedDict, total=False):
    step_idx: Required[int]
    walker_idx: Required[int]
    decision_id: Required[int]
    target_idxs: Required[tuple[int, ...]]

WARPING_RECORD_FIELDS = frozenset({
    "walker_idx",
    "target_idx",
    "weight",
})

@attrs.define
class WarpRecord:
    walker_idx: int
    target_idx: int
    weight: float

class WarpRecordUnstruct(TypedDict, total=False):
    walker_idx: Required[int]
    target_idx: Required[int]
    weight: Required[float]

