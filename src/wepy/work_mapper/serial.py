"""Reference implementation of a serial WorkMapper"""

# Standard Library
import logging
import time
from typing import (
    Callable,
    Generic,
    TypeVar,
)

# First Party Library
from wepy.factory import Factory
from wepy.runners.runner import RunSegmentData
from wepy.walker import WalkerState

logger = logging.getLogger(__name__)


WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
RunSegmentData_ = TypeVar("RunSegmentData_", bound=RunSegmentData)


class SerialMapper(
    Generic[
        WalkerState_,
        RunSegmentData_,
    ]
):
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
            WalkerState_,
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


class SerialMapperFactory(Factory[SerialMapper]):

    @classmethod
    def type(cls) -> type[SerialMapper]:
        return SerialMapper

    def __call__(self) -> SerialMapper:
        return SerialMapper()
