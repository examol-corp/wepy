import itertools
import multiprocessing as mp
from typing import Any, Callable, Literal
import logging

import attrs

from wepy.work_mapper.base import AnyWalkerState

logger = logging.getLogger(__name__)

class ProcPoolMapper:

    num_workers: int

    def __init__(
            self,
            num_workers: int,
            proc_start_method: Literal["fork", "spawn", "forkserver"] = "spawn",
    ) -> None:
        self.num_workers = num_workers
        self.proc_start_method = proc_start_method

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

        logger.info("Initializing ProcPoolMapper")

        self._func = segment_func

        logger.info(f"Initializing local multiprocessing context with start method: {self.proc_start_method}")
        self._mp_ctx = mp.get_context(method=self.proc_start_method)

    def cleanup(self) -> None:

        logger.info("Running ProcPoolMapper cleanup")
        logger.info("Nothing to do")

    def map(
            self,
            walker_states: list[AnyWalkerState],
            *args: list[list[Any]],
    ) -> list[AnyWalkerState]:

        logger.info(f"Running map on {len(walker_states)} in batches of {self.num_workers}")

        # spin up a new pool for each map
        logger.info(f"Starting process Pool with {self.num_workers}")
        with self._mp_ctx.Pool(
                processes=self.num_workers,
                # only run one thing per task, just to make sure
                # everything is cleaned up
                maxtasksperchild=1,
        ) as pool:

            results = []
            for batch_idx, batch in enumerate(itertools.batched(
                    zip(walker_states, *args, strict=True),
                    self.num_workers,
                    strict=False,
            )):

                logger.info(f"Submitting batch: {batch_idx}")

                batch_results = []
                for batch_task_idx, batch_args in enumerate(batch):

                    task_idx = batch_idx + batch_task_idx

                    logger.info(f"Submitting task {task_idx}")
                    result = pool.apply_async(
                        self._func,
                        batch_args,
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
