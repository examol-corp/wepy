# Standard Library
import time
from typing import Callable

# Third Party Library
import attrs

# First Party Library
from wepy.runners.openmm import (
    OpenMMPlatformName,
    OpenMMState,
)
from wepy.work_mapper.base import WorkMapper


class OpenMMSerialWorkMapper(WorkMapper):

    def __init__(
        self,
        platform: OpenMMPlatformName,
        global_platform_properties: dict[str, str] | None = None,
    ) -> None:
        self._worker_segment_times: dict[int, list[float]] = {0: []}

        self._platform = platform
        self._global_platform_properties = (
            global_platform_properties if global_platform_properties is not None else {}
        )

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
        for task_idx, task_args in enumerate(
            zip(walker_states, segment_lengths, strict=True)
        ):

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
