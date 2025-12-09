"""Base classes and definitions for all work mappers."""

# Standard Library
import traceback
from typing import Callable, Literal, Generic, TypeVar, Protocol, Any, ParamSpec, Concatenate
import logging

# Standard Library

from wepy.walker import WalkerState
from wepy.runners.runner import RunSegmentData

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
RunSegmentData_ = TypeVar("RunSegmentData_", bound=RunSegmentData)

class WorkMapper(Protocol[WalkerState_, RunSegmentData_]):

    def init(self) -> None:
        ...

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
