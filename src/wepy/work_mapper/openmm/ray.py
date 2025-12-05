"""Special OpenMM mappers."""
import logging
from typing import Literal, Any, Callable
import time
import itertools

import attrs
import ray
import ray.util.multiprocessing

from wepy.work_mapper.base import WorkMapper
from wepy.runners.openmm import OpenMMState, OpenMMRunner, PlatformKwargs, OpenMMPlatformName

logger = logging.getLogger(__name__)

@attrs.define
class OpenMMRayTask:

    openmm_task: OpenMMTask

    def __call__(
            self,
            *args,
            **kwargs,
    ) -> OpenMMState:

        logging.basicConfig(level=logging.INFO)
        logging.getLogger("OpenMMRayTask").info("Configured logging in OpenMMRayTask process")

        return self.openmm_task(*args, **kwargs)


class OpenMMRayPoolWorkMapper:

    
    def __init__(
        self,
        platform: str,
        device_ids: list[int] | None = None,
        device_platform_properties: list[dict[str, str]] | None = None,
        global_platform_properties: dict[str, str] | None = None,
        ray_init_args: dict[str,Any] | None = None,
    ):

        if platform in {"CUDA", "HIP", "OpenCL"}:
            if device_ids is None:
                raise ValueError(f"For accelerator platforms ({platform} requested) device_ids must be given.")

        if device_platform_properties is not None:

            if len(device_platform_properties) != device_ids:
                raise ValueError(f"{len(device_ids)} requested, but only {len(device_platform_properties)} given.")

            else:
                self._device_platform_properties = {
                    idx : props
                    for idx, props
                    in enumerate(device_platform_properties)
                }


        else:
            self._device_platform_properties = None

        self._platform = platform
        self._device_ids: dict[int,int] = {
            idx: device_id
            for idx, device_id
            in enumerate(device_ids)
        }
        self._num_workers = len(device_ids)

        if ray_init_args is not None:
            self._ray_init_args = ray_init_args

        else:
            self._ray_init_args = None


        self._global_platform_properties = global_platform_properties

    def init(
            self,
    ) -> None:

        logger.info("Initializing RayPoolMapper")

        self._ray_ctx = ray.init(**(self._ray_init_args if self._ray_init_args is not None else {}))


    def cleanup(self) -> None:

        logger.info("Running RayPoolMapper cleanup")

        ray.shutdown()

    def map(
            self,
            tasks: list[OpenMMTask],
            walker_states: list[OpenMMState],
    ) -> list[OpenMMState]:

        logger.info(f"Running map on {len(walker_states)} in batches of {self._num_workers}")

        # spin up a new pool for each map
        logger.info(f"Starting ray Pool with {self._num_workers} workers")
        with ray.util.multiprocessing.Pool(
                processes=self._num_workers,
                # only run one thing per task, just to make sure
                # everything is cleaned up
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

                    if self._device_ids is not None and self._platform in GPU_PLATFORMS:
                        logger.info("Worker platform configured and resolving platform properties.")
                        platform_kwargs = self._global_platform_properties | {
                            "DeviceIndex" : str(self._device_ids[worker_idx]),
                            **(
                                self._device_platform_properties[worker_idx]
                                if self._device_platform_properties is not None
                                else {}
                            ),
                        }
                    else:
                        logger.info("Non-worker platform, only using global properties")
                        platform_kwargs = self._global_platform_properties

                    _task = OpenMMRayTask(task)
                    logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
                    logger.info(f"Injecting: platform={self._platform}, platform_kwargs={platform_kwargs}")
                    result = pool.apply_async(
                        _task,
                        args=(walker_state,),
                        kwargs=dict(
                            platform=self._platform,
                            platform_kwargs=platform_kwargs
                        )
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
