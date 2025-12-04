"""Special OpenMM mappers."""
import logging
from typing import Literal, Any, Callable
import time
import multiprocessing as mp
import itertools

import attrs
import ray
import ray.util.multiprocessing

from wepy.work_mapper.base import WorkMapper
from wepy.runners.openmm import OpenMMState, OpenMMRunner, PlatformKwargs, OpenMMPlatformName

logger = logging.getLogger(__name__)

GPU_PLATFORMS = {"CUDA", "OpenCL", "HIP"}

class OpenMMSerialWorkMapper(WorkMapper):

    def __init__(
        self,
        platform: OpenMMPlatformName,
        global_platform_properties: dict[str, str] | None = None,
    ) -> None:
        self._worker_segment_times: dict[int, list[float]] = {0: []}

        self._platform = platform
        self._global_platform_properties = global_platform_properties if global_platform_properties is not None else {}

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
            task: Callable[[OpenMMState, int], OpenMMState],
            walker_states: list[OpenMMState],
            segment_lengths: list[int],
    ) -> list[OpenMMState]:

        segment_times: list[float] = []
        results: list[OpenMMState] = []
        for task_idx, task_args in enumerate(zip(walker_states, segment_lengths, strict=True)):

            tic = time.time()
            result = task(
                *task_args,
                platform_name=self._platform,
                platform_kwargs=self._global_platform_properties,
            )
            toc = time.time()

            segment_times.append(toc - tic)
            results.append(result)

        self._worker_segment_times[0] = segment_times

        return results

@attrs.define
class OpenMMSerialWorkMapperFactory:

    platform: OpenMMPlatformName
    global_platform_properties: dict[str, str] | None = None

    def __call__(self) -> OpenMMSerialWorkMapper:

        return OpenMMSerialWorkMapper(
            platform=self.platform,
            global_platform_properties=self.global_platform_properties,
        )
    

class OpenMMProcPoolWorkMapper(WorkMapper):

    def __init__(
        self,
        platform: OpenMMPlatformName,
        num_procs: int,
        device_ids: list[int] | None = None,
        global_platform_properties: dict[str, str] | None = None,
        device_platform_properties: list[dict[str, str]] | None = None,
        proc_start_method: Literal["fork", "spawn", "forkserver"] = "spawn",
    ):


        if platform in {"CUDA", "HIP", "OpenCL"}:
            if device_ids is None:
                raise ValueError(f"For accelerator platforms ({platform} requested) device_ids must be given.")

        if device_platform_properties is not None:

            if len(device_platform_properties) != len(device_ids):
                raise ValueError(f"{len(device_ids)} devices requested, but only {len(device_platform_properties)} device platform property dicts given.")

            else:
                self._device_platform_properties = {
                    idx : props
                    for idx, props
                    in enumerate(device_platform_properties)
                }


        else:
            self._device_platform_properties = None

        self._platform = platform
        self._global_platform_properties = global_platform_properties

        if device_ids is not None and num_procs != len(device_ids):

            raise ValueError(f"When device_ids is given ({device_ids}) it must be the same length as the number of processes: {num_procs}")
        
        self._device_ids: dict[int,int] = {
            idx: device_id
            for idx, device_id
            in enumerate(device_ids)
        } if device_ids is not None else None
        self._num_procs = num_procs

        self._proc_start_method = proc_start_method
        
    def init(
            self,
    ) -> None:

        logger.info("Initializing ProcPoolMapper")

        logger.info(f"Initializing local multiprocessing context with start method: {self._proc_start_method}")
        self._mp_ctx = mp.get_context(method=self._proc_start_method)

    def cleanup(self) -> None:

        logger.info("Running ProcPoolMapper cleanup")
        logger.info("Nothing to do")
            

    def map(
            self,
            task: Callable[[OpenMMState, int], OpenMMState],
            walker_states: list[OpenMMState],
            segment_lengths: list[int],
    ) -> list[OpenMMState]:

        logger.info(f"Running map on {len(walker_states)} in batches of {self._num_procs}")

        # spin up a new pool for each map
        logger.info(f"Starting process Pool with {self._num_procs}")
        with self._mp_ctx.Pool(
                processes=self._num_procs,
                # only run one thing per task, just to make sure
                # everything is cleaned up
                maxtasksperchild=1,
        ) as pool:

            results = []
            for batch_idx, batch in enumerate(itertools.batched(
                    zip(walker_states, segment_lengths, strict=True),
                    self._num_procs,
                    strict=False,
            )):

                logger.info(f"Submitting batch: {batch_idx}")

                batch_results = []
                for batch_task_idx, task_args in enumerate(batch):

                    task_idx = batch_idx + batch_task_idx
                    # for our purposes each element in this batch
                    # should be associated with a worker.
                    worker_idx = batch_task_idx

                    if self._device_ids is not None:
                        logger.info("device_ids have been given, setting up special platform properties for each task.")
                        device_id = str(self._device_ids[worker_idx])
                        worker_platform_props = {
                            **(
                                {"DeviceIndex" : device_id}
                                if self._platform in GPU_PLATFORMS
                                else {}
                            ),
                            **(
                                self._device_platform_properties[worker_idx]
                                if (
                                        self._device_platform_properties is not None and
                                        worker_idx in self._device_platform_properties
                                )
                                else {}
                            )
                        }
                        logger.info(
                            f"Device IDs given, resolved to using worker specific platform properties: {worker_platform_props}"
                        )
                    else:
                        logger.info("No Device IDs given.")
                        worker_platform_props = None

                    match (self._global_platform_properties, worker_platform_props):
                        case (None, None):
                            logger.info("No platform properties provided")
                            _platform_kwargs = {}
                        case (global_kwargs, None):
                            logger.info("Only global platform properties provided")
                            _platform_kwargs = global_kwargs
                        case (None, local_kwargs):
                            logger.info("Only device specific platform properties provided")
                            _platform_kwargs = worker_platform_props
                        case (global_kwargs, local_kwargs):
                            logger.info("Both global and device specific platform properties provided")
                            _platform_kwargs = global_kwargs | worker_platform_props

                    logger.info(f"Resolved 'platform_kwargs' : {_platform_kwargs}")

                    logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
                    result = pool.apply_async(
                        task,
                        args=task_args,
                        kwds=dict(
                            platform_name=self._platform,
                            platform_kwargs=_platform_kwargs,
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

@attrs.define
class OpenMMProcPoolWorkMapperFactory:

    platform: OpenMMPlatformName
    num_procs: int
    device_ids: list[int] | None = None
    global_platform_properties: dict[str, str] | None = None
    device_platform_properties: list[dict[str, str]] | None = None
    proc_start_method: Literal["fork", "spawn", "forkserver"] = "spawn"

    def __attrs_post_init__(self) -> None:

        if self.platform in {"CUDA", "HIP", "OpenCL"}:
            if self.device_ids is None:
                raise ValueError(f"For accelerator platforms ({self.platform} requested) device_ids must be given.")

        if self.device_platform_properties is not None:

            if len(self.device_platform_properties) != len(self.device_ids):
                raise ValueError(f"{len(self.device_ids)} devices requested, but only {len(self.device_platform_properties)} device platform property dicts given.")

        if self.device_ids is not None and self.num_procs != len(self.device_ids):

            raise ValueError(f"When device_ids is given ({self.device_ids}) it must be the same length as the number of processes: {self.num_procs}")
            

    def __call__(self) -> OpenMMProcPoolWorkMapper:

        return OpenMMProcPoolWorkMapper(
            platform=self.platform,
            num_procs=self.num_procs,
            device_ids=self.device_ids,
            global_platform_properties=self.global_platform_properties,
            device_platform_properties=self.device_platform_properties,
            proc_start_method=self.proc_start_method,
        )


# @attrs.define
# class OpenMMRayTask:

#     openmm_task: OpenMMTask

#     def __call__(
#             self,
#             *args,
#             **kwargs,
#     ) -> OpenMMState:

#         logging.basicConfig(level=logging.INFO)
#         logging.getLogger("OpenMMRayTask").info("Configured logging in OpenMMRayTask process")

#         return self.openmm_task(*args, **kwargs)


# class OpenMMRayPoolWorkMapper:

    
#     def __init__(
#         self,
#         platform: str,
#         device_ids: list[int] | None = None,
#         device_platform_properties: list[dict[str, str]] | None = None,
#         global_platform_properties: dict[str, str] | None = None,
#         ray_init_args: dict[str,Any] | None = None,
#     ):

#         if platform in {"CUDA", "HIP", "OpenCL"}:
#             if device_ids is None:
#                 raise ValueError(f"For accelerator platforms ({platform} requested) device_ids must be given.")

#         if device_platform_properties is not None:

#             if len(device_platform_properties) != device_ids:
#                 raise ValueError(f"{len(device_ids)} requested, but only {len(device_platform_properties)} given.")

#             else:
#                 self._device_platform_properties = {
#                     idx : props
#                     for idx, props
#                     in enumerate(device_platform_properties)
#                 }


#         else:
#             self._device_platform_properties = None

#         self._platform = platform
#         self._device_ids: dict[int,int] = {
#             idx: device_id
#             for idx, device_id
#             in enumerate(device_ids)
#         }
#         self._num_workers = len(device_ids)

#         if ray_init_args is not None:
#             self._ray_init_args = ray_init_args

#         else:
#             self._ray_init_args = None


#         self._global_platform_properties = global_platform_properties

#     def init(
#             self,
#     ) -> None:

#         logger.info("Initializing RayPoolMapper")

#         self._ray_ctx = ray.init(**(self._ray_init_args if self._ray_init_args is not None else {}))


#     def cleanup(self) -> None:

#         logger.info("Running RayPoolMapper cleanup")

#         ray.shutdown()

#     def map(
#             self,
#             tasks: list[OpenMMTask],
#             walker_states: list[OpenMMState],
#     ) -> list[OpenMMState]:

#         logger.info(f"Running map on {len(walker_states)} in batches of {self._num_workers}")

#         # spin up a new pool for each map
#         logger.info(f"Starting ray Pool with {self._num_workers} workers")
#         with ray.util.multiprocessing.Pool(
#                 processes=self._num_workers,
#                 # only run one thing per task, just to make sure
#                 # everything is cleaned up
#                 maxtasksperchild=1,
#         ) as pool:

#             results = []
#             for batch_idx, batch in enumerate(itertools.batched(
#                     zip(walker_states, tasks, strict=True),
#                     self._num_workers,
#                     strict=False,
#             )):

#                 logger.info(f"Submitting batch: {batch_idx}")

#                 batch_results = []
#                 for batch_task_idx, (walker_state, task) in enumerate(batch):

#                     task_idx = batch_idx + batch_task_idx
#                     # for our purposes each element in this batch
#                     # should be associated with a worker.
#                     worker_idx = batch_task_idx

#                     if self._device_ids is not None and self._platform in GPU_PLATFORMS:
#                         logger.info("Worker platform configured and resolving platform properties.")
#                         platform_kwargs = self._global_platform_properties | {
#                             "DeviceIndex" : str(self._device_ids[worker_idx]),
#                             **(
#                                 self._device_platform_properties[worker_idx]
#                                 if self._device_platform_properties is not None
#                                 else {}
#                             ),
#                         }
#                     else:
#                         logger.info("Non-worker platform, only using global properties")
#                         platform_kwargs = self._global_platform_properties

#                     _task = OpenMMRayTask(task)
#                     logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
#                     logger.info(f"Injecting: platform={self._platform}, platform_kwargs={platform_kwargs}")
#                     result = pool.apply_async(
#                         _task,
#                         args=(walker_state,),
#                         kwargs=dict(
#                             platform=self._platform,
#                             platform_kwargs=platform_kwargs
#                         )
#                     )
#                     logger.info(f"Task {task_idx} submitted")
#                     batch_results.append(result)

#                 logger.info(f"Batch {batch_idx} submitted, awaiting results.")
#                 for batch_task_idx, task_result in enumerate(batch_results):

#                     task_idx = batch_idx + batch_task_idx
#                     logger.info(f"Awaiting task {task_idx}")


#                     try:
#                         real_result = task_result.get()
#                     # TODO: add timeouts and retries
#                     except TimeoutError as exc:
#                         raise exc
#                     except Exception as exc:
#                         raise exc

#                     results.append(real_result)

#                     logger.info(f"Retrieved completed results for task: {task_idx}")

#                 logger.info(f"Batch {batch_idx} completed")

#             logger.info(f"Completed all batches, terminating Pool")


#         return results
