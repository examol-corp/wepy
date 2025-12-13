"""Defines a generic protocol for storage of data.

This should only define the interfaces between the generating
components (resamplers, runner, boundary conditions, sim_manager) and
the reporting and storage backends should utilize.

"""
from collections.abc import Mapping
from numpy.typing import NDArray

RecordValueDtype = int | float | NDArray
Record = Mapping[str, RecordValueDtype]
