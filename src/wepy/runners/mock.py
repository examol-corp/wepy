"""Realistic mock runners useful mostly for testing."""
import logging

import attrs
from wepy.interface import Task, RunnerGenTaskArgs
from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner
from wepy.work_mapper.base import Task

logger = logging.getLogger(__name__)

@attrs.define
class MockState(WalkerState):
    a: int

class MockError(Exception):
    pass

@attrs.define
class MockTask(Task):

    runner: "MockRunner"
    segment_length: int
    fail: bool

    def __call__(self, state: MockState) -> MockState:
        logger.info("Running MockTask segment")
        return self.runner.run_segment(state, self.segment_length, self.fail)

@attrs.define
class MockRunner(Runner):

    fail_walker_idxs: set[int] = attrs.field(default={})

    def pre_cycle(self) -> None:
        pass

    def post_cycle(self) -> None:
        pass

    def gen_tasks(self, segment_spec: RunnerGenTaskArgs[MockState]) -> list[MockTask]:

        return [
            MockTask(
                runner=self,
                segment_length=segment_spec.segment_length,
                fail=(
                    True
                    if walker_idx in self.fail_walker_idxs
                    else False
                )
            )
            for walker_idx, state
            in enumerate(segment_spec.states)
        ]

    def run_segment(
        self,
        state: MockState,
        segment_length: int,
        fail: bool,
    ) -> MockState:

        if fail:
            logger.critical("Error requested in MockRuner.run_segment, raising.")
            raise MockError("Error requested")

        logger.info("Evolving the MockState in MockRunner.run_segment")
        return attrs.evolve(
            state,
            a=(state.a + segment_length),
        )

    def get_last_cycle_segments_split_times(self) -> None:
        return None

