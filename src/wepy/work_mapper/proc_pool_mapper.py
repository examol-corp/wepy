import itertools
import multiprocessing as mp
from typing import Any, Callable, Literal, Generic, TypeVar, Never
import logging

import attrs

from wepy.work_mapper.base import Task, WalkerState, WorkMapper

# log_safe.initialize_safe_logging()

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
Task_ = TypeVar("Task_", bound=Task)

def _force_error(exc: Exception) -> Never:
    raise exc

@attrs.define
class ProcPoolTask(Task, Generic[Task_]):

    wrapped_task: Task_

    def __call__(self, walker_state: WalkerState_) -> WalkerState_:

        logging.basicConfig(level="INFO")
        logging.getLogger("ProcPoolTask").info("Configured logging in task process")

        return self.wrapped_task(walker_state)

class ProcPoolMapper(
        WorkMapper,
        Generic[
            WalkerState_,
            Task_,
        ],
):

    def __init__(
        self,
        num_workers: int,
    ) -> None:
        self._num_workers = num_workers

    def init(
            self,
    ) -> None:
        """l..

        worker_func_args: This is an arbitrary set of key-values that
            for each worker will be passed into the segment_func call
            if that worker is used. Useful for injecting things like
            device IDs.

        """

        logger.info("Initializing ProcPoolMapper")

        logger.info(f"Initializing local multiprocessing context with 'spawn' process start method")

        # NOTE: always require "spawn" as this is the safest and
        # changing to "fork" can have lots of other effects that we
        # don't want to test
        self._mp_ctx = mp.get_context(method="spawn")

    def cleanup(self) -> None:

        logger.info("Running ProcPoolMapper cleanup")
        logger.info("Nothing to do")

    def map(
            self,
            tasks: list[Task_],
            walker_states: list[WalkerState_],
    ) -> list[WalkerState_]:

        logger.info(f"Running map on {len(walker_states)} in batches of {self._num_workers}")

        # spin up a new pool for each map
        logger.info(f"Starting process Pool with {self._num_workers}")

        with self._mp_ctx.Pool(
                processes=self._num_workers,
                # NOTE: only run one thing per task, just to make sure
                # everything is cleaned up which is an issue with
                # OpenMM contexts. Also note that this is why we use a
                # multiprocessing.Pool and not a
                # concurrent.futures.ProcessPoolExecutor
                maxtasksperchild=1,
        ) as pool:

            results = []
            for batch_idx, batch in enumerate(itertools.batched(
                    zip(walker_states, tasks, strict=True),
                    self._num_workers,
                    strict=False,
            )):

                logger.info(f"Submitting batch: {batch_idx}")

                batch_results = []
                for batch_task_idx, (walker_state, task) in enumerate(batch):

                    task_idx = batch_idx + batch_task_idx
                    # for our purposes each element in this batch
                    # should be associated with a worker.
                    worker_idx = batch_task_idx

                    proc_pool_task = ProcPoolTask(task)

                    logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
                    result = pool.apply_async(
                        proc_pool_task,
                        (walker_state,),
                        # NOTE: this must be provided or in some cases when a
                        # worker crashes on startup it will hang
                        error_callback=_force_error,
                    )
                    logger.info(f"Task {task_idx} submitted")
                    batch_results.append(result)

                logger.info(f"Batch {batch_idx} submitted, awaiting results.")
                for batch_task_idx, task_result in enumerate(batch_results):

                    task_idx = batch_idx + batch_task_idx
                    logger.info(f"Awaiting task {task_idx}")


                    try:
                        real_result = task_result.get()
                    # TODO: add timeouts and retries
                    except TimeoutError as exc:
                        raise exc
                    except Exception as exc:
                        raise exc

                    results.append(real_result)

                    logger.info(f"Retrieved completed results for task: {task_idx}")

                logger.info(f"Batch {batch_idx} completed")

            logger.info(f"Completed all batches, terminating Pool")


        return results
