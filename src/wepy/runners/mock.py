"""Realistic mock runners useful mostly for testing."""
import logging
from typing import Literal

import attrs
from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner

logger = logging.getLogger(__name__)

@attrs.define
class MockState(WalkerState):
    a: int

class MockError(Exception):
    pass

@attrs.define
class MockRunner(Runner):

    fail: bool = False

    _initialized: bool = False

    def init(self) -> None:
        self._initialized = True

    def pre_cycle(self) -> None:
        pass
    def post_cycle(self) -> None:
        pass

    def run_segment(
        self,
        state: MockState,
        segment_length: int,
    ) -> MockState:

        if self.fail:
            logger.critical("Error requested in MockRuner.run_segment, raising.")
            raise MockError("Error requested")

        logger.info("Evolving the MockState in MockRunner.run_segment")
        return attrs.evolve(
            state,
            a=(state.a + segment_length),
        )

    def get_last_cycle_segments_split_times(self) -> None:
        return None
