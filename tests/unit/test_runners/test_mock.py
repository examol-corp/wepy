# Third Party Library
import pytest

# First Party Library
from wepy.runners.mock import MockRunner, MockState
from wepy.runners.runner import (
    RunnerStateMachineError,
    RunnerStateTransitionError,
    RunnerStatus,
)


class TestMockRunner:

    def test_init(self):

        runner = MockRunner()
        assert runner.status == RunnerStatus.PRE_INITIALIZATION
        runner.init()
        assert runner.status == RunnerStatus.INITIALIZED

        with pytest.raises(RunnerStateTransitionError):
            runner.init()

    def test_pre_cycle(self):
        runner = MockRunner()

        with pytest.raises(RunnerStateTransitionError):
            runner.pre_cycle()

        runner.init()
        runner.pre_cycle()
        assert runner.status == RunnerStatus.PRE_CYCLE

        with pytest.raises(RunnerStateTransitionError):
            runner.pre_cycle()

        runner.post_cycle(None)

        assert runner.status == RunnerStatus.POST_CYCLE
        runner.pre_cycle()
        assert runner.status == RunnerStatus.PRE_CYCLE

    def test_post_cycle(self):
        runner = MockRunner()

        with pytest.raises(RunnerStateTransitionError):
            runner.pre_cycle()

        runner.init()
        runner.pre_cycle()

        with pytest.raises(RunnerStateTransitionError):
            runner.pre_cycle()

        runner.post_cycle(None)
        runner.pre_cycle()
        runner.post_cycle(None)

    def test_run_segment(self):

        runner = MockRunner()

        with pytest.raises(RunnerStateMachineError):
            runner.run_segment(
                MockState(0),
                10,
            )

        runner.init()
        runner.pre_cycle()

        assert runner.run_segment(
            MockState(0),
            10,
        )[
            0
        ] == MockState(10)
