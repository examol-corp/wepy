"""Realistic mock runners useful mostly for testing."""
import logging

import attrs
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
class MockRunner(Runner):

    def pre_cycle(self) -> None:
        pass

    def post_cycle(self) -> None:
        pass

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

@attrs.define
class MockTask(Task):

    runner: MockRunner
    segment_length: int
    fail: bool

    def __call__(self, state: MockState) -> MockState:
        logger.info("Running MockTask segment")
        return self.runner.run_segment(state, self.segment_length, self.fail)
