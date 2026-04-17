# Standard Library
import copy
import logging

# Third Party Library
import attrs
import numpy as np
import openmm
import openmm.app
import openmm.unit
import pytest

# First Party Library
from wepy.runners.openmm.logger import StepIntervalLoggingReporter
from wepy.runners.openmm.runner import (
    _DEFAULT_HEARTBEAT_INTERVAL,
    _DEFAULT_STATE_TIME_INTERVAL,
    OpenMMRunner,
    OpenMMRunnerFactory,
    OpenMMRunnerSegmentData,
)
from wepy.runners.openmm.state import (
    OpenMMState,
    dummy_context,
)
from wepy.runners.runner import (
    RunnerStateError,
    RunnerStateTransitionError,
    RunnerStatus,
)
from wepy_tools.systems.lennard_jones import LennardJonesPair

UNIT_CUBE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

STEP_SIZE = 2 * openmm.unit.femtoseconds


@pytest.fixture
def runner_components() -> (
    tuple[openmm.System, openmm.app.Topology, openmm.LangevinIntegrator]
):

    lj_sys = LennardJonesPair()

    integrator = openmm.LangevinIntegrator(300.0, 0.1, STEP_SIZE)

    return lj_sys.system, lj_sys.topology, integrator


@pytest.fixture
def omm_context() -> openmm.Context:

    lj_sys = LennardJonesPair()

    ctx = dummy_context(lj_sys.system, lj_sys.positions)

    return ctx


@attrs.define
class Spy:
    touched: bool = False

    def touch(self) -> None:
        self.touched = True


class TouchGlobalStepIntervalLoggingReporter(StepIntervalLoggingReporter):

    def __init__(
        self,
        logger: logging.Logger,
        start_time: int,
        spy: Spy,
    ) -> None:

        self.spy = spy

        super().__init__(
            logger=logger,
            callback=self.touch,
            state_includes=[],
            # NOTE: hardcoded
            step_interval=1,
            start_time=start_time,
        )

    def touch(self, *args) -> None:
        self.spy.touch()


class Test_OpenMMRunner:

    def test___init__(self, runner_components):

        system, topology, integrator = runner_components

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
            openmm_reporter_factories=None,
        )
        assert runner.openmm_reporter_factories == []
        assert runner._openmm_reporters is None
        assert runner._init_time is None

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )
        assert runner.openmm_reporter_factories == []

        assert runner.status == RunnerStatus.PRE_INITIALIZATION

        SPY = Spy()

        def _mock_factory(
            logger: logging.Logger, start_time: int
        ) -> TouchGlobalStepIntervalLoggingReporter:
            return TouchGlobalStepIntervalLoggingReporter(
                logger=logger, start_time=start_time, spy=SPY
            )

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
            openmm_reporter_factories=[_mock_factory],
        )
        assert len(runner.openmm_reporter_factories) == 1

    def test_init(self, runner_components):

        system, topology, integrator = runner_components
        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )

        assert runner._openmm_reporters is None

        runner.init()
        assert runner.status == RunnerStatus.INITIALIZED

        # check the default openmm reporters were constructed
        assert runner._init_time is not None

        # test status, can't init twice
        with pytest.raises(RunnerStateTransitionError):
            runner.init()

    def test_pre_cycle(self, runner_components):
        system, topology, integrator = runner_components

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )

        with pytest.raises(RunnerStateTransitionError):
            runner.pre_cycle()

        runner.init()
        assert runner.status == RunnerStatus.INITIALIZED

        runner.pre_cycle()
        assert runner.status == RunnerStatus.PRE_CYCLE
        assert runner._pre_cycle_time is not None

    def test_run_segment(self, runner_components):

        system, topology, integrator = runner_components

        lj_sys = LennardJonesPair()

        state = OpenMMState.from_dwim(positions=lj_sys.positions)

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )

        with pytest.raises(RunnerStateError):
            runner.run_segment(state, 2)

        runner.init()
        with pytest.raises(RunnerStateError):
            runner.run_segment(state, 2)

        runner.pre_cycle()

        new_state, segment_data = runner.run_segment(state, 2)

        assert "positions" in new_state
        assert "velocities" in new_state

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
            get_state_keys={
                "positions",
            },
        )
        runner.init()
        runner.pre_cycle()

        new_state, segment_data = runner.run_segment(
            new_state,
            2,
        )
        assert new_state["positions"] is not None
        assert "velocities" not in new_state

        assert isinstance(segment_data, OpenMMRunnerSegmentData)

        # test that openmm reporters are being called
        SPY = Spy()

        def _mock_factory(
            logger: logging.Logger, start_time: int
        ) -> TouchGlobalStepIntervalLoggingReporter:
            return TouchGlobalStepIntervalLoggingReporter(
                logger=logger, start_time=start_time, spy=SPY
            )

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
            openmm_reporter_factories=[_mock_factory],
        )
        runner.init()
        runner.pre_cycle()

        assert not SPY.touched
        new_state, segment_data = runner.run_segment(
            state,
            10,
        )

        assert SPY.touched

    def test_post_cycle(self, runner_components):

        system, topology, integrator = runner_components

        lj_sys = LennardJonesPair()

        state = OpenMMState.from_dwim(positions=lj_sys.positions)

        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )

        with pytest.raises(RunnerStateTransitionError):
            runner.post_cycle(None)

        runner.init()
        with pytest.raises(RunnerStateTransitionError):
            runner.post_cycle(None)

        runner.pre_cycle()

        # NOTE: that you don't need to call run_segment, because in a
        # standard use case this would be done in a subprocess. Any
        # state changes must be reified in the RunSegmentData

        runner.post_cycle(None)

        assert runner.status == RunnerStatus.POST_CYCLE

        # with a run_segment
        runner = OpenMMRunner(
            system=copy.deepcopy(system),
            topology=copy.deepcopy(topology),
            integrator=copy.deepcopy(integrator),
        )

        with pytest.raises(RunnerStateTransitionError):
            runner.post_cycle(None)

        runner.init()
        with pytest.raises(RunnerStateTransitionError):
            runner.post_cycle(None)

        runner.pre_cycle()
        new_state, segment_data = runner.run_segment(
            state,
            10,
        )

        runner.post_cycle([segment_data])

        assert runner.status == RunnerStatus.POST_CYCLE


def test_OpenMMRunnerFactory(runner_components):

    system, topology, integrator = runner_components

    # NOTE: no need to copy at this level because the factory handles
    # that for you
    omm_factory = OpenMMRunnerFactory(
        system=system,
        topology=topology,
        integrator=integrator,
    )

    assert isinstance(omm_factory(), OpenMMRunner)


def test_defaults(runner_components):
    """A test that exercises the default settings of the logging reporters."""

    system, topology, integrator = runner_components

    lj_sys = LennardJonesPair()

    state = OpenMMState.from_dwim(positions=lj_sys.positions)

    runner = OpenMMRunnerFactory(
        system=system,
        topology=topology,
        integrator=integrator,
    )()

    time_interval_steps = _DEFAULT_STATE_TIME_INTERVAL / STEP_SIZE

    _steps = (
        time_interval_steps
        if time_interval_steps > _DEFAULT_HEARTBEAT_INTERVAL
        else _DEFAULT_HEARTBEAT_INTERVAL
    )

    runner.init()
    runner.pre_cycle()
    new_state, segment_data = runner.run_segment(state, 2 * _steps)
