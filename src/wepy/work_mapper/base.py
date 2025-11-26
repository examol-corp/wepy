"""Base classes and definitions for all work mappers."""

# Standard Library
import traceback
from typing import Callable, Literal, Generic, TypeVar, Protocol, Any
import logging

# Standard Library

from wepy.walker import WalkerState

logger = logging.getLogger(__name__)

AnyWalkerState = TypeVar("AnyWalkerState", bound=WalkerState)


class WorkMapper(Protocol[AnyWalkerState]):

    def init(
            self,
            segment_func: Callable[
                tuple[
                    AnyWalkerState,
                    ...,
                ],
                AnyWalkerState,
            ]
    ) -> None:
        ...

    def map(
        self,
        walker_states: list[AnyWalkerState],
        *args: list[list[Any]],
        **kwargs: dict[str, list[Any]],
    ) -> list[AnyWalkerState]: ...

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

class Task:
    """Class that composes a function and arguments."""

    def __init__(self, func, *args, **kwargs):
        """Constructor for Task.

        Parameters
        ----------
        func : callable
            Function to be called on the arguments.

        *args
            The arguments to pass to func

        """
        self.args = args
        self.kwargs = kwargs
        self.func = func

    def __call__(self, **worker_kwargs):
        """Makes the Task itself callable."""

        # run the function passing in the args for running it and any
        # worker information in the worker kwargs.
        return self.func(*self.args, **self.kwargs, **worker_kwargs)


# class ABCMapper:
#     """Abstract base class for a Mapper."""

#     def __init__(
#             self,
#             segment_func: SegmentFunc | None =None,
#             **kwargs: dict[str, Any],
#     ) -> None:
#         """Constructor for the Mapper class. No arguments are required.

#         Parameters
#         ----------

#         segment_func : Set a default segment_func. Typically set at
#           runtime.

#         """

#         self._func = segment_func

#         self._attributes = kwargs

#     @property
#     def attributes(self) -> dict[str, Any]:
#         return self._attributes

#     def init(
#             self,
#             segment_func: SegmentFunc | None = None,
#             **kwargs: dict[str, Any],
#     ) -> None:
#         """Runtime initialization and setting of function to map over walkers.

#         Parameters
#         ----------
#         segment_func : callable implementing the Runner.run_segment interface

#         """

#         if self.segment_func is not None and segment_func is not None:
#             logger.info(
#                 "overriding default segment_func {} with {}".format(
#                     self._func, segment_func
#                 )
#             )
#             self._func = segment_func

#         elif self.segment_func is None and segment_func is None:
#             ValueError("segment_func must be given since no default specified")

#         elif self.segment_func is None and segment_func is not None:
#             self._func = segment_func

#     @property
#     def segment_func(self) -> SegmentFunc:
#         """The function that will be called for new data in the `map` method."""
#         return self._func

#     def cleanup(self, **kwargs: dict[str, Any]) -> None:
#         """Runtime post-simulation tasks.

#         This is run either at the end of a successful simulation or
#         upon an error in the main process of the simulation manager
#         call to `run_cycle`.

#         The Mapper class performs no actions here and all arguments
#         are ignored.

#         """

#         # nothing to do
#         pass

#     def map(self, walkers: list[Walker], *args: list[list[Any]], **kwargs: dict[list[Any]]) -> list[Walker]:
#         raise NotImplementedError
