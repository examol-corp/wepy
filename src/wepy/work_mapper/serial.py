"""Reference implementation of a serial WorkMapper"""
import time
import sys
import traceback

from typing import Callable, Literal, Generic, TypeVar, Protocol, Any, ParamSpec, Concatenate, Sequence
import logging

from wepy.walker import Walker, WalkerState
from wepy.runners.runner import RunSegmentData

logger = logging.getLogger(__name__)


WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
RunSegmentData_ = TypeVar("RunSegmentData_", bound=RunSegmentData)

class SerialMapper(
        Generic[
            WalkerState_,
            RunSegmentData_,
        ]):
    """Basic non-parallel reference implementation of a mapper."""

    def __init__(
        self,
    ) -> None:
        self._worker_segment_times: dict[int, list[float]] = {0: []}


    def get_worker_segment_times(self) -> dict[int, list[float]]:
        """The run timings for each segment for each walker.

        Returns
        -------
        worker_seg_times : Dictionary mapping worker indices to a list of times in
            seconds for each segment run.

        """
        return self._worker_segment_times

    def init(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def map(
            self,
            task: Callable[
                [
                    WalkerState_,
                    int,
                ],
                WalkerState_
            ],
            walker_states: list[WalkerState_],
            segment_lengths: list[int],
    ) -> list[tuple[WalkerState_, RunSegmentData_]]:
        segment_times: list[float] = []
        results: list[WalkerState_] = []
        for task_idx, task_args in enumerate(
                zip(
                    walker_states,
                    segment_lengths,
                    strict=True,
                )
        ):

            tic = time.time()
            result = task(*task_args)
            toc = time.time()

            segment_times.append(toc - tic)
            results.append(result)

        self._worker_segment_times[0] = segment_times

        return results
