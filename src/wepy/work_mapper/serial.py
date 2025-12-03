"""Reference implementation of a serial WorkMapper"""
import time
import sys
import traceback

from typing import Callable, Literal, Generic, TypeVar, Protocol, Any, ParamSpec, Concatenate, Sequence
import logging

from wepy.walker import Walker, WalkerState
from wepy.work_mapper.base import WorkMapper, TaskException, Task

logger = logging.getLogger(__name__)


WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
Task_ = TypeVar("Task_", bound=Task)

class SerialMapper(
        WorkMapper,
        Generic[
            WalkerState_,
            Task_,
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

    def gen_task(
            self,
            outer_task: Task_,
            task_idx: int,
    ) -> Task_:
        return outer_task

    def map(
            self,
            tasks: list[Task_],
            walker_states: list[WalkerState_],
    ) -> list[WalkerState_]:
        segment_times: list[float] = []
        results: list[WalkerState_] = []
        for task_idx, (task, walker_state) in enumerate(zip(tasks, walker_states, strict=True)):

            _task = self.gen_task(task, task_idx=task_idx)

            tic = time.time()
            result = _task(walker_state)
            toc = time.time()

            segment_times.append(toc - tic)
            results.append(result)

        self._worker_segment_times[0] = segment_times

        return results
