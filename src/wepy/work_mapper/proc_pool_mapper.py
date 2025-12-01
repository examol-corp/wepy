import itertools
import multiprocessing as mp
from typing import Any, Callable, Literal
import logging

import attrs

from wepy.work_mapper.base import AnyWalkerState

logger = logging.getLogger(__name__)

class ProcPoolMapper:

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
            num_workers: int,
            worker_args: list[dict[str, Any]] | None = None,
            proc_start_method: Literal["fork", "spawn", "forkserver"] = "spawn",
    ) -> None:
        """l..

        worker_func_args: This is an arbitrary set of key-values that
            for each worker will be passed into the segment_func call
            if that worker is used. Useful for injecting things like
            device IDs.

        """

        logger.info("Initializing ProcPoolMapper")

        self._num_workers = num_workers
        self._proc_start_method = proc_start_method

        if worker_args is not None and len(worker_args) != num_workers:
            raise ValueError("If worker_args are given they must match the number of workers.")

        elif worker_args is None:
            logger.info("No worker arguments given.")
            self._worker_args = [{} for _ in range(self._num_workers)]

        else:
            logger.info(f"Configured workers with the following function arguments: {worker_args}")
            self._worker_args = worker_args
            

        self._func = segment_func

        logger.info(f"Initializing local multiprocessing context with start method: {self._proc_start_method}")
        self._mp_ctx = mp.get_context(method=self._proc_start_method)

    def cleanup(self) -> None:

        logger.info("Running ProcPoolMapper cleanup")
        logger.info("Nothing to do")

    def map(
            self,
            walker_states: list[AnyWalkerState],
            *args: list[list[Any]],
    ) -> list[AnyWalkerState]:

        logger.info(f"Running map on {len(walker_states)} in batches of {self._num_workers}")

        # spin up a new pool for each map
        logger.info(f"Starting process Pool with {self._num_workers}")
        with self._mp_ctx.Pool(
                processes=self._num_workers,
                # only run one thing per task, just to make sure
                # everything is cleaned up
                maxtasksperchild=1,
        ) as pool:

            results = []
            for batch_idx, batch in enumerate(itertools.batched(
                    zip(walker_states, *args, strict=True),
                    self._num_workers,
                    strict=False,
            )):

                logger.info(f"Submitting batch: {batch_idx}")

                batch_results = []
                for batch_task_idx, batch_args in enumerate(batch):

                    task_idx = batch_idx + batch_task_idx
                    # for our purposes each element in this batch
                    # should be associated with a worker.
                    worker_idx = batch_task_idx

                    logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
                    result = pool.apply_async(
                        self._func,
                        args=batch_args,
                        kwds=self._worker_args[worker_idx],
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
