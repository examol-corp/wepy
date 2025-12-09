import pytest
import attrs
from wepy.runners.runner import NoRunner, RunnerStateMachine, RunnerStatus, RunnerEvent, RunnerStateTransitionError, RunnerStateError
from wepy.walker import Walker, WalkerState

class Test_RunnerStateMachine:

    def test_validate_event(self):

        sm = RunnerStateMachine(state=RunnerStatus.PRE_INITIALIZATION)
        assert sm.validate_event(RunnerEvent.INIT)

        sm = RunnerStateMachine(state=RunnerStatus.PRE_INITIALIZATION)
        with pytest.raises(RunnerStateTransitionError):
            sm.validate_event(RunnerEvent.PRE_CYCLE)
        
    def test_send(self):

        sm = RunnerStateMachine(state=RunnerStatus.PRE_INITIALIZATION)
        # make sure the return and state are consistent
        assert sm.send(RunnerEvent.INIT) == RunnerStatus.INITIALIZED
        assert sm.state == RunnerStatus.INITIALIZED

        # default construction
        assert RunnerStateMachine().state == RunnerStatus.PRE_INITIALIZATION

        sm = RunnerStateMachine(state=RunnerStatus.PRE_INITIALIZATION)
        with pytest.raises(RunnerStateTransitionError):
            sm.send(RunnerEvent.PRE_CYCLE)

        # test the rest of the transitions
        assert RunnerStateMachine(
            RunnerStatus.INITIALIZED
        ).send(RunnerEvent.PRE_CYCLE) == RunnerStatus.PRE_CYCLE

        assert RunnerStateMachine(
            RunnerStatus.PRE_CYCLE
        ).send(RunnerEvent.POST_SEGMENT) == RunnerStatus.POST_SEGMENT

        assert RunnerStateMachine(
            RunnerStatus.POST_SEGMENT
        ).send(RunnerEvent.POST_CYCLE) == RunnerStatus.POST_CYCLE

        assert RunnerStateMachine(
            RunnerStatus.POST_CYCLE
        ).send(RunnerEvent.PRE_CYCLE) == RunnerStatus.PRE_CYCLE

class Test_NoRunner:

    def test___init__(self):

        assert NoRunner().state_machine.state == RunnerStatus.PRE_INITIALIZATION

    def test_status(self):
        assert NoRunner().status == RunnerStatus.PRE_INITIALIZATION

    def test_init(self):
        runner = NoRunner()
        runner.init()
        assert runner.status == RunnerStatus.INITIALIZED

    def test_pre_cycle(self):
        runner = NoRunner()
        runner.init()
        runner.pre_cycle()
        assert runner.status == RunnerStatus.PRE_CYCLE

    def test_post_cycle(self):
        runner = NoRunner()
        runner.init()
        runner.pre_cycle()
        runner.post_cycle(None)
        assert runner.status == RunnerStatus.POST_CYCLE

    def test_run_segment(self):

        # concrete state to use
        @attrs.define
        class SomeState(WalkerState):
            a: int

            def __getitem__(self, key: str) -> int:

                if key != "a":
                    raise KeyError(f"Invalid key '{key}'")

                return self.a

            def dict(self) -> dict[str, int]:
                return {"a" : self.a}

        runner = NoRunner()

        walker = Walker(
            state=SomeState(a=1),
            weight=0.1,
        )

        with pytest.raises(RunnerStateError):
            runner.run_segment(
                walker,
                10,
            )
        
        runner.init()

        with pytest.raises(RunnerStateError):
            runner.run_segment(
                walker,
                10,
            )
        
        runner.pre_cycle()
        assert (
            runner.run_segment(
                walker,
                10,
            )
            == (walker, None)
        )

        runner.post_cycle(None)
        with pytest.raises(RunnerStateError):
            runner.run_segment(
                walker,
                10,
            )
        
