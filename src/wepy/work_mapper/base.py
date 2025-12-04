"""Base classes and definitions for all work mappers."""

# Standard Library
import traceback
from typing import Callable, Literal, Generic, TypeVar, Protocol, Any, ParamSpec, Concatenate
import logging

# Standard Library

from wepy.walker import WalkerState

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)

class WorkMapper(Protocol[WalkerState_]):

    def init(self) -> None:
        ...

    def map(
        self,
        task: Callable[[WalkerState_, int], WalkerState_],
        walker_states: list[WalkerState_],
        segment_lengths: list[int],
    ) -> list[WalkerState_]: ...

    def get_worker_segment_times(self) -> dict[int, list[float]] | None: ...

    def cleanup(self) -> None: ...
