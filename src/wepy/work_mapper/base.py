"""Base classes and definitions for all work mappers."""

# Standard Library
import traceback
from typing import Callable, Literal, Generic, TypeVar, Protocol, Any, ParamSpec, Concatenate
import logging

# Standard Library

from wepy.interface import (
    WorkMapperFactoryArgs,
    Task,
)
from wepy.walker import WalkerState

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
Task_ = TypeVar("Task_", bound=Task)


    
class WorkMapper(Protocol[WalkerState_, Task_]):

    def __init__(
        self,
        segment_func: Callable[
            [
                WalkerState_,
                Task_,
            ],
            WalkerState_,
        ],
        wm_args: WorkMapperFactoryArgs | None,
    ) -> None:
        ...

    def init(self) -> None:
        ...

    def map(
        self,
        tasks: list[Task_],
        walker_states: list[WalkerState_],
    ) -> list[WalkerState_]: ...

    def get_worker_segment_times(self) -> dict[int, list[float]] | None: ...

    def cleanup(self) -> None: ...

class WrapperException(Exception):
    """Exception used for wrapping another exception.

    Since tracebacks can't be pickled we format it and save that
    instead.

    """

    def __init__(
        self,
        message,
        # must be kwargs so we can pickle it (I know weird...)
        wrapped_exception=None,
        tb=None,
    ):
        super().__init__(message)

        # save the exception with the traceback
        self.wrapped_exception = wrapped_exception
        self.formatted_tb = traceback.format_tb(tb)

class TaskException(WrapperException):
    pass


