"""Realistic mock runners useful mostly for testing."""

# Standard Library
import logging
import time

# Third Party Library
import attrs

# First Party Library
from wepy.runners.runner import (
    Runner,
    RunnerEvent,
    RunnerStateError,
    RunnerStateMachine,
    RunnerStatus,
    RunSegmentData,
)
from wepy.walker import WalkerState, AttrsWalkerStateMixin

logger = logging.getLogger(__name__)


@attrs.define
class MockState(AttrsWalkerStateMixin, WalkerState):
    a: int


class MockError(Exception):
    pass


@attrs.define
class MockRunner(Runner):
    fail: bool = False

    state_machine: RunnerStateMachine = attrs.field(
        default=attrs.Factory(
            RunnerStateMachine,
        )
    )

    @property
    def status(self) -> RunnerStatus:
        return self.state_machine.state

    def init(self) -> None:

        # NOTE: showing example of validating the event before doing
        # potentially expensive calculations and then transitioning
        # the actual state when it is done
        self.state_machine.validate_event(RunnerEvent.INIT)
        # do something...
        logger.info("INIT stuff")
        self.state_machine.send(RunnerEvent.INIT)

    def pre_cycle(self) -> None:
        self.state_machine.send(RunnerEvent.PRE_CYCLE)

    def post_cycle(self, segments_data: list[RunSegmentData]) -> None:
        self.state_machine.send(RunnerEvent.POST_SEGMENT)
        self.state_machine.send(RunnerEvent.POST_CYCLE)

    def run_segment(
        self,
        state: MockState,
        segment_length: int,
    ) -> tuple[MockState, RunSegmentData]:

        if self.status != RunnerStatus.PRE_CYCLE:
            raise RunnerStateError(
                f"Cannot run a segment in state ({self.status.name}:{self.status.value})"
            )

        seg_start_time = time.time()

        if self.fail:
            logger.critical("Error requested in MockRunner.run_segment, raising.")
            raise MockError("Error requested")

        logger.info("Evolving the MockState in MockRunner.run_segment")
        new_state = attrs.evolve(
            state,
            a=(state.a + segment_length),
        )

        seg_end_time = time.time()

        split_time = seg_end_time - seg_start_time

        segment_data = RunSegmentData(segment_split_time=split_time)

        return new_state, segment_data


@attrs.define
class MockRunnerFactory:
    fail: bool = False

    def __call__(self) -> MockRunner:
        return MockRunner(fail=self.fail)
