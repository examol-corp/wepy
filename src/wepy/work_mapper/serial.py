"""Reference implementation of a serial WorkMapper"""
import time
import sys
import traceback

from typing import Callable, Literal, Generic, TypeVar, Protocol, Any
import logging

from wepy.walker import Walker
from wepy.work_mapper.base import WorkMapper, AnyWalkerState, TaskException

logger = logging.getLogger(__name__)



class SerialMapper(WorkMapper[AnyWalkerState]):
    """Basic non-parallel reference implementation of a mapper."""

    def __init__(
        self,
    ) -> None:
        """Constructor for the Mapper class. No arguments are required."""

        self._worker_segment_times: dict[int, list[float]] = {0: []}


    def get_worker_segment_times(self) -> dict[int, list[float]]:
        """The run timings for each segment for each walker.

        Returns
        -------
        worker_seg_times : Dictionary mapping worker indices to a list of times in
            seconds for each segment run.

        """
        return self._worker_segment_times
        
    def init(
            self,
            segment_func: Callable[
                [
                    AnyWalkerState,
                    Any,
                    ...,
                ],
                AnyWalkerState,
            ],
    ) -> None:

        self._func = segment_func

    def cleanup(self) -> None:
        pass

    def map(
            self,
            walker_states: list[AnyWalkerState],
            *args: list[list[Any]],
            **kwargs: dict[str, list[Any]],
    ) -> list[AnyWalkerState]:
        """Map the 'segment_func' to args.

        Parameters
        ----------
        *args : list of list
            Each element is the argument to one call of 'segment_func'.

        Returns
        -------
        results : list
            The results of each call to 'segment_func' in the same order as input.

        Examples
        --------
        >>> Mapper(segment_func=sum).map([(0,1,2), (3,4,5)])
        [3, 12]

        """

        segment_times: list[float] = []
        results: list[AnyWalkerState] = []

        for arg_idx, (walker_state, *call_args) in enumerate(zip(walker_states, *args, strict=True)):

            call_kwargs = {
                k : values[arg_idx]
                for k, values
                in kwargs.items()
            }

            tic = time.time()
            result = self._func(walker_state, *call_args, **call_kwargs)
            toc = time.time()

            segment_times.append(toc - tic)
            results.append(result)

        self._worker_segment_times[0] = segment_times

        return results
