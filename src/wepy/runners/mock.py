"""Realistic mock runners useful mostly for testing."""

import attrs
from wepy.walker import Walker
from wepy.runners.runner import Runner

@attrs.define
class MockState:
    a: 1

class MockError(Exception):
    pass

@attrs.define
class MockRunner(Runner[MockState]):

    fail: bool = False

    def pre_cycle(self) -> None:
        pass

    def post_cycle(self) -> None:
        pass

    def run_segment(
        self,
        state: MockState,
        segment_length: int,
        worker_id: int = 0,
        # UGLY: here to satisfy the interface
        cycle_idx: int = 0,
        walker_idx: int = 0
    ) -> MockState:

        if self.fail:
            raise MockError("Error requested")

        return attrs.evolve(
            state,
            a=(state.a + segment_length),
        )

    def get_last_cycle_segments_split_times(self) -> None:
        return None
