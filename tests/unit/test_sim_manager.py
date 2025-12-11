# Standard Library
import copy

# Third Party Library
import pytest

# First Party Library
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.runners.mock import MockError, MockRunnerFactory, MockState
from wepy.runners.runner import RunnerStatus, RunSegmentData
from wepy.sim_manager import (
    Manager,
    ManagerEvent,
    ManagerStateMachine,
    ManagerStateTransitionError,
    ManagerStatus,
)
from wepy.walker import Walker
from wepy.work_mapper.serial import SerialMapper


@pytest.fixture
def sim_components() -> tuple[
    list[Walker],
    MockRunnerFactory,
    type[NoResampler],
]:

    num_walkers = 4

    init_walker_weight = 1 / num_walkers
    init_walkers = [
        Walker(
            state=MockState(1),
            weight=init_walker_weight,
        )
        for walker_state in range(num_walkers)
    ]

    return init_walkers, MockRunnerFactory(fail=False), NoResampler


class Test_ManagerStateMachine:

    def test_validate_event(self):

        sm = ManagerStateMachine()
        assert sm.validate_event(ManagerEvent.START_PRE_SIM)

        sm = ManagerStateMachine()
        with pytest.raises(ManagerStateTransitionError):
            sm.validate_event(ManagerEvent.FINISH_SIMULATION)

    def test_send(self):

        sm = ManagerStateMachine(state=ManagerStatus.CONSTRUCTED)
        # make sure the return and state are consistent
        assert sm.send(ManagerEvent.START_PRE_SIM) == ManagerStatus.PRE_SIMULATION
        assert sm.state == ManagerStatus.PRE_SIMULATION

        # default construction
        assert ManagerStateMachine().state == ManagerStatus.CONSTRUCTED

        # TODO: this should be its own test as it is a behavioural test

        # Test all the transitions are what we expect
        # sm = ManagerStateMachine(state=ManagerStatus.PRE_INITIALIZATION)


class Test_Manager:

    def test___init__(self, sim_components):

        manager = Manager(*sim_components)

        assert manager.status == ManagerStatus.CONSTRUCTED
        assert len(manager.reporters) == 0
        assert manager.work_mapper_factory == SerialMapper
        assert manager._work_mapper is None

    def test_init(self, sim_components):

        manager = Manager(*sim_components)

        with pytest.raises(ManagerStateTransitionError):
            manager.init()

        manager.state_machine.send(ManagerEvent.START_PRE_SIM)

        manager.init()
        assert manager.status == ManagerStatus.INITIALIZED
        assert manager._work_mapper is not None
        assert manager._runner.status == RunnerStatus.INITIALIZED

        with pytest.raises(ManagerStateTransitionError):
            manager.init()

    def test_pre_segment(self, sim_components):

        manager = Manager(*sim_components)

        with pytest.raises(ManagerStateTransitionError):
            manager.pre_segment()

        manager.state_machine.send(ManagerEvent.START_PRE_SIM)

        manager.init()

        manager.state_machine.send(ManagerEvent.START_SIM)
        manager.state_machine.send(ManagerEvent.START_CYCLE)

        manager.pre_segment()

        assert manager.status == ManagerStatus.PRE_SEGMENT_FINISHED
        assert manager._runner.status == RunnerStatus.PRE_CYCLE

    def test_post_segment(self, sim_components):

        manager = Manager(*sim_components)

        with pytest.raises(ManagerStateTransitionError):
            manager.post_segment(None)

        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()
        manager.state_machine.send(ManagerEvent.START_SIM)
        manager.state_machine.send(ManagerEvent.START_CYCLE)
        manager.pre_segment()

        manager.run_segment(
            [copy.deepcopy(walker.state) for walker in sim_components[0]],
            5,
            0,
        )

        manager.post_segment(
            [
                RunSegmentData(
                    segment_split_time=2.0,
                )
                for _ in range(len(sim_components[0]))
            ]
        )

        assert manager.status == ManagerStatus.POST_SEGMENT_FINISHED
        assert manager._runner.status == RunnerStatus.POST_CYCLE

    def test_cleanup(self, sim_components):

        manager = Manager(*sim_components)

        # cleanup can be run in any state after init, but not after cleanup

        with pytest.raises(ManagerStateTransitionError):
            manager.cleanup()

        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()
        assert manager._runner.status == RunnerStatus.INITIALIZED
        manager.cleanup()

        with pytest.raises(ManagerStateTransitionError):
            manager.cleanup()

    def test_run_segment(self, sim_components):

        init_walkers, runner_factory, resampler = sim_components

        manager = Manager(*sim_components)
        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()

        # UGLY,TODO: that this is ugly as there might be some implicit
        # state around that needs to go in tandem. So there should be
        # some state coupled data that should be introduced to make
        # this more robust

        # all the state changes needed for this to work
        manager.state_machine.send(ManagerEvent.START_SIM)
        manager.state_machine.send(ManagerEvent.START_CYCLE)
        manager.pre_segment()

        new_states = manager.run_segment(
            [walker.state for walker in init_walkers],
            1,
            0,
        )
        assert manager.status == ManagerStatus.SEGMENT_FINISHED

        # test if something fails
        manager = Manager(
            init_walkers,
            MockRunnerFactory(fail=True),
            resampler,
        )
        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()
        manager.state_machine.send(ManagerEvent.START_SIM)
        manager.state_machine.send(ManagerEvent.START_CYCLE)
        manager.pre_segment()

        with pytest.raises(MockError):
            manager.run_segment(
                [walker.state for walker in init_walkers],
                1,
                0,
            )

        assert manager.status == ManagerStatus.CLEANUP_FINISHED

    def test_run_cycle(self, sim_components):

        init_walkers, runner, resampler = sim_components

        manager = Manager(*sim_components)
        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()
        manager.state_machine.send(ManagerEvent.START_SIM)

        new_states = manager.run_cycle(
            init_walkers,
            1,
            0,
        )

        assert manager.status == ManagerStatus.POST_CYCLE

        # test if something fails
        manager = Manager(
            init_walkers,
            MockRunnerFactory(fail=True),
            resampler,
        )
        manager.state_machine.send(ManagerEvent.START_PRE_SIM)
        manager.init()
        manager.state_machine.send(ManagerEvent.START_SIM)

        with pytest.raises(MockError):
            manager.run_cycle(
                init_walkers,
                1,
                0,
            )

        assert manager.status == ManagerStatus.CLEANUP_FINISHED

    def test_run_simulation(self, sim_components):

        manager = Manager(*sim_components)

        new_walkers, _ = manager.run_simulation(2, 2)
        assert manager.status == ManagerStatus.FINISHED

        manager = Manager(*sim_components)

        new_walkers, _ = manager.run_simulation(
            2,
            2,
            continue_run_idx=0,
        )
        assert manager.status == ManagerStatus.FINISHED

    def test_run_simulation_by_time(self, sim_components):

        manager = Manager(*sim_components)

        new_walkers, _ = manager.run_simulation_by_time(
            0.001,
            2,
        )
        assert manager.status == ManagerStatus.FINISHED

        manager = Manager(*sim_components)
        new_walkers, _ = manager.run_simulation_by_time(
            0.001,
            2,
            continue_run_idx=0,
        )
        assert manager.status == ManagerStatus.FINISHED

        # make sure it runs at least one cycle
        manager = Manager(*sim_components)
        new_walkers, _ = manager.run_simulation_by_time(
            0.0000001,
            2,
            continue_run_idx=0,
        )
        assert manager.status == ManagerStatus.FINISHED
