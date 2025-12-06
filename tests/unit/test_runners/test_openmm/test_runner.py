import logging
from wepy_tools.systems.lennard_jones import LennardJonesPair
from wepy.runners.openmm.state import (
    dummy_context,
    OpenMMState,
)

from wepy.runners.openmm.runner import (
    OpenMMRunner,
)

from wepy.runners.openmm.logger import StepIntervalLoggingReporter
import pytest

import numpy as np
import openmm
import openmm.app
import openmm.unit

UNIT_CUBE = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

@pytest.fixture
def runner_components() -> tuple[openmm.System, openmm.app.Topology, openmm.LangevinIntegrator]:

    lj_sys = LennardJonesPair()

    integrator  = openmm.LangevinIntegrator(300.0, 0.002, 0.1)

    return lj_sys.system, lj_sys.topology, integrator

@pytest.fixture
def omm_context() -> openmm.Context:

    lj_sys = LennardJonesPair()

    ctx = dummy_context(lj_sys.system, lj_sys.positions)

    return ctx



class TestOpenMMRunner:

    def test___init__(self, runner_components):

        runner = OpenMMRunner(
            *runner_components
        )

    def test_init(self, runner_components):

        runner = OpenMMRunner(
            *runner_components
        )

        assert runner._openmm_reporters == []

        runner.init()

        # check the default openmm reporters were constructed
        assert len(runner._openmm_reporters) > 0


    def test_pre_cycle(self, runner_components):

        runner = OpenMMRunner(
            *runner_components,
        )

        assert runner._last_cycle_segments_split_times == []
        runner.pre_cycle()
        assert runner._last_cycle_segments_split_times == []

        runner = OpenMMRunner(
            *runner_components,
            last_cycle_segments_split_times=[{"something" : 1}]
        )

        runner.pre_cycle()
        assert runner._last_cycle_segments_split_times == []

    def test_run_segment(self, runner_components):

        system, topology, integrator = runner_components

        lj_sys = LennardJonesPair()

        state = OpenMMState.from_dwim(positions=lj_sys.positions)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            platform_name="Reference",
        )

        new_state = runner.run_segment(state, 2)
        assert "positions" in new_state
        assert "velocities" in new_state

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            get_state_keys={"positions",}
        )

        new_state = runner.run_segment(
            new_state,
            2,
        )
        assert new_state["positions"] is not None
        assert "velocities" not in new_state

        # test that openmm reporters are being called
        class Spy:
            def __init__(self) -> None:
                self.touched = False

            def touch(self) -> None:
                self.touched = True

        SPY = Spy()

        class TouchGlobalStepIntervalLoggingReporter(StepIntervalLoggingReporter):

            def __init__(
                self,
                logger: logging.Logger,
            ) -> None:

                self.spy = SPY

                super().__init__(
                    logger=logger,
                    callback=self.touch,
                    state_includes=[],
                    # NOTE: hardcoded
                    step_interval=1,
                )

            def touch(self, *args) -> None:
                self.spy.touch()


        def _mock_factory(logger: logging.Logger) -> TouchGlobalStepIntervalLoggingReporter:
            return TouchGlobalStepIntervalLoggingReporter(logger=logger)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            get_state_keys={},
            openmm_reporter_factories=[_mock_factory],
        )
        runner.init()

        assert not SPY.touched
        new_state = runner.run_segment(
            state,
            2,
        )

        assert SPY.touched
