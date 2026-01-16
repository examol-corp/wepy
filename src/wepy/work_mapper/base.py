"""Base classes and definitions for all work mappers."""

# Standard Library
import logging
from typing import (
    Callable,
    Protocol,
    TypeVar,
)

# First Party Library
from wepy.runners.runner import RunSegmentData
from wepy.walker import WalkerState

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
RunSegmentData_ = TypeVar("RunSegmentData_", bound=RunSegmentData)


class WorkMapper(Protocol[WalkerState_, RunSegmentData_]):

    def init(self) -> None: ...

    def map(
        self,
        task: Callable[[WalkerState_, int], tuple[WalkerState_, RunSegmentData_]],
        walker_states: list[WalkerState_],
        segment_lengths: list[int],
    ) -> list[
        tuple[
            WalkerState_,
            RunSegmentData_,
        ]
    ]: ...

    def get_worker_segment_times(self) -> dict[int, list[float]] | None: ...

    def cleanup(self) -> None: ...

