"""Special OpenMM mappers."""

# Standard Library
import itertools
import logging
import multiprocessing as mp
from typing import Callable

# Third Party Library
import attrs

# First Party Library
import more_itertools
from wepy.runners.openmm import (
    GPU_PLATFORMS,
    OpenMMPlatformName,
    OpenMMState,
)
from wepy.util.multiprocessing import proc_pool_worker_setup, queue_listener_context
from wepy.work_mapper.base import WorkMapper

logger = logging.getLogger(__name__)


class OpenMMProcPoolWorkMapper(WorkMapper):

    def __init__(
        self,
        platform: OpenMMPlatformName,
        num_procs: int,
        device_ids: list[int] | None = None,
        global_platform_properties: dict[str, str] | None = None,
        device_platform_properties: list[dict[str, str]] | None = None,
    ):

        if platform in {"CUDA", "HIP", "OpenCL"}:
            if device_ids is None:
                raise ValueError(
                    f"For accelerator platforms ({platform} requested) device_ids must be given."
                )

        if device_platform_properties is not None:

            if len(device_platform_properties) != len(device_ids):
                raise ValueError(
                    f"{len(device_ids)} devices requested, but only {len(device_platform_properties)} device platform property dicts given."
                )

            else:
                self._device_platform_properties = {
                    idx: props for idx, props in enumerate(device_platform_properties)
                }

        else:
            self._device_platform_properties = None

        self._platform = platform
        self._global_platform_properties = global_platform_properties

        if device_ids is not None and num_procs != len(device_ids):

            raise ValueError(
                f"When device_ids is given ({device_ids}) it must be the same length as the number of processes: {num_procs}"
            )

        self._device_ids: dict[int, int] = (
            {idx: device_id for idx, device_id in enumerate(device_ids)}
            if device_ids is not None
            else None
        )
        self._num_procs = num_procs

    def init(
        self,
    ) -> None:

        logger.info("Initializing ProcPoolMapper")

        logger.info(
            "Initializing local multiprocessing context with start method: spawn"
        )
        self._mp_ctx = mp.get_context(method="spawn")

    def cleanup(self) -> None:

        logger.info("Running ProcPoolMapper cleanup")
        logger.info("Nothing to do")

    def map(
        self,
        task: Callable[[OpenMMState, int], OpenMMState],
        walker_states: list[OpenMMState],
        segment_lengths: list[int],
    ) -> list[OpenMMState]:

        logger.info(
            f"Running map on {len(walker_states)} in batches of {self._num_procs}"
        )

        # spin up a new pool for each map
        logger.info(f"Starting process Pool with {self._num_procs}")

        with (
            queue_listener_context(self._mp_ctx) as log_queue,
            self._mp_ctx.Pool(
                processes=self._num_procs,
                # only run one thing per task, just to make sure
                # everything is cleaned up
                maxtasksperchild=1,
                initializer=proc_pool_worker_setup,
                initargs=(log_queue,),
            ) as pool,
        ):

            results = []
            for batch_idx, batch in enumerate(
                more_itertools.chunked(
                    zip(walker_states, segment_lengths, strict=True),
                    self._num_procs,
                    strict=False,
                )
            ):

                logger.info(f"Submitting batch: {batch_idx}")

                batch_results = []
                for batch_task_idx, task_args in enumerate(batch):

                    task_idx = batch_idx + batch_task_idx
                    # for our purposes each element in this batch
                    # should be associated with a worker.
                    worker_idx = batch_task_idx

                    if self._device_ids is not None:
                        logger.info(
                            "device_ids have been given, setting up special platform properties for each task."
                        )
                        device_id = str(self._device_ids[worker_idx])
                        worker_platform_props = {
                            **(
                                {"DeviceIndex": device_id}
                                if self._platform in GPU_PLATFORMS
                                else {}
                            ),
                            **(
                                self._device_platform_properties[worker_idx]
                                if (
                                    self._device_platform_properties is not None
                                    and worker_idx in self._device_platform_properties
                                )
                                else {}
                            ),
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
                            logger.info(
                                "Only device specific platform properties provided"
                            )
                            _platform_kwargs = worker_platform_props
                        case (global_kwargs, local_kwargs):
                            logger.info(
                                "Both global and device specific platform properties provided"
                            )
                            _platform_kwargs = global_kwargs | worker_platform_props

                    logger.info(f"Resolved 'platform_kwargs' : {_platform_kwargs}")

                    logger.info(f"Submitting task {task_idx} to worker {worker_idx}")
                    result = pool.apply_async(
                        task,
                        args=task_args,
                        kwds=dict(
                            platform_name=self._platform,
                            platform_kwargs=_platform_kwargs,
                        ),
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

            logger.info("Completed all batches, terminating Pool")

        return results


@attrs.define
class OpenMMProcPoolWorkMapperFactory:

    platform: OpenMMPlatformName
    num_procs: int
    device_ids: list[int] | None = None
    global_platform_properties: dict[str, str] | None = None
    device_platform_properties: list[dict[str, str]] | None = None

    def __attrs_post_init__(self) -> None:

        if self.platform in {"CUDA", "HIP", "OpenCL"}:
            if self.device_ids is None:
                raise ValueError(
                    f"For accelerator platforms ({self.platform} requested) device_ids must be given."
                )

        if self.device_platform_properties is not None:

            if len(self.device_platform_properties) != len(self.device_ids):
                raise ValueError(
                    f"{len(self.device_ids)} devices requested, but only {len(self.device_platform_properties)} device platform property dicts given."
                )

        if self.device_ids is not None and self.num_procs != len(self.device_ids):

            raise ValueError(
                f"When device_ids is given ({self.device_ids}) it must be the same length as the number of processes: {self.num_procs}"
            )

    @classmethod
    def type(cls) -> type[OpenMMProcPoolWorkMapper]:
        return OpenMMProcPoolWorkMapper
        

    def __call__(self) -> OpenMMProcPoolWorkMapper:

        return OpenMMProcPoolWorkMapper(
            platform=self.platform,
            num_procs=self.num_procs,
            device_ids=self.device_ids,
            global_platform_properties=self.global_platform_properties,
            device_platform_properties=self.device_platform_properties,
        )
