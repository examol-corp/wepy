# Standard Library
import copy
import logging
import time

# Third Party Library
import openmm
import openmm.app
import openmm.unit
import pytest

# First Party Library
from wepy.runners.openmm.logger import (
    EnergyLoggingReporter,
    HeartBeatLoggingReporter,
    LoggingReporter,
    SamplingTimeIntervalLoggingReporter,
    StepIntervalLoggingReporter,
    UnitCellLoggingReporter,
)
from wepy.runners.openmm.reporter import OpenMMReporterNextReport
from wepy.runners.openmm.state import OpenMMState
from wepy_tools.systems.lennard_jones import LennardJonesPair

STEP_TIME = 1 * openmm.unit.femtosecond


@pytest.fixture(scope="function")
def sim_components() -> tuple[
    openmm.State,
    openmm.app.Topology,
    openmm.System,
    openmm.LangevinIntegrator,
    openmm.Platform,
]:

    lj_sys = LennardJonesPair()
    integrator = openmm.VerletIntegrator(STEP_TIME)
    omm_state = (
        OpenMMState.from_dwim(positions=lj_sys.positions).to_state_wrapper().state
    )

    platform = openmm.Platform.getPlatformByName("Reference")

    return omm_state, lj_sys.topology, lj_sys.system, integrator, platform


class Test_LoggingReporter:

    def test_report(self, caplog):

        logger = logging.getLogger("test-LoggingReporter")

        def hello_log(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> None:

            logger.info("Hello")

        hello_log_reporter = LoggingReporter(
            logger,
            callback=hello_log,
            state_includes=["energy"],
        )

        # NOTE: dummy inputs since they aren't used in the hello_log callback
        with caplog.at_level(logging.INFO, logger="test-LoggingReporter"):
            hello_log_reporter.report(None, None)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[0].msg == "Hello"

        caplog.clear()


class Test_StepIntervalLoggingReporter:

    def test_describeNextReport(self, sim_components):

        omm_state, sim_args = sim_components[0], sim_components[1:]

        logger = logging.getLogger("test-StepIntervalLoggingReporter")

        def hello_log(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> None:

            logger.info("Hello")

        state_includes = ["energy"]

        step_logger = StepIntervalLoggingReporter(
            logger,
            callback=hello_log,
            state_includes=state_includes,
            step_interval=10,
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(*sim_args)
        simulation.context.setState(omm_state)

        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=10,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(1)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=9,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(2)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=7,
            include=list(state_includes),
            periodic=False,
        )

        # wraps back around at 0
        simulation.step(7)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=10,
            include=list(state_includes),
            periodic=False,
        )

    def test_simulation(self, sim_components, caplog):
        omm_state, sim_args = sim_components[0], sim_components[1:]

        simulation = openmm.app.Simulation(*sim_args)
        simulation.context.setState(omm_state)

        logger_name = "test-StepIntervalLoggingReporter"
        logger = logging.getLogger(logger_name)

        def hello_log(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> None:

            logger.info("Hello")

        state_includes = ["energy"]

        step_logger = StepIntervalLoggingReporter(
            logger,
            callback=hello_log,
            state_includes=state_includes,
            step_interval=10,
            start_time=time.time(),
        )

        simulation.reporters.append(step_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(1)

        assert len(caplog.records) == 0
        caplog.clear()

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(9)

        assert len(caplog.records) == 1
        assert caplog.records[0].msg == "Hello"

        caplog.clear()


class Test_SamplingTimeIntervalLoggingReporter:

    def test_describeNextReport(self, sim_components):

        omm_state, sim_args = sim_components[0], sim_components[1:]
        topology, system, integrator, platform = sim_args

        logger = logging.getLogger("test-SamplingTimeIntervalLoggingReporter")

        def hello_log(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> None:

            logger.info("Hello")

        state_includes = ["energy"]

        step_logger = SamplingTimeIntervalLoggingReporter(
            logger,
            callback=hello_log,
            state_includes=state_includes,
            sampling_time_interval=(10 * openmm.unit.femtosecond),
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)

        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=10,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(1)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=9,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(2)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=7,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(7)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=10,
            include=list(state_includes),
            periodic=False,
        )

        simulation.step(3)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=7,
            include=list(state_includes),
            periodic=False,
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)

        simulation.step(10)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=10,
            include=list(state_includes),
            periodic=False,
        )

        # test wrapping around behavior
        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)

        simulation.step(11)
        assert step_logger.describeNextReport(simulation) == OpenMMReporterNextReport(
            steps=9,
            include=list(state_includes),
            periodic=False,
        )

    def test_simulation(self, sim_components, caplog):
        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-SamplingTimeIntevalLoggingReporter"
        logger = logging.getLogger(logger_name)

        def hello_log(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
        ) -> None:

            logger.info(f"Step: {state.getStepCount()}")

        state_includes = ["energy"]

        time_logger = SamplingTimeIntervalLoggingReporter(
            logger,
            callback=hello_log,
            state_includes=state_includes,
            sampling_time_interval=(10 * openmm.unit.femtosecond),
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        simulation.reporters.append(time_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(1)

        assert len(caplog.records) == 0
        logger.info("Next test")
        caplog.clear()

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(9)

        assert len(caplog.records) == 1
        assert caplog.records[0].msg == "Step: 10"

        logger.info("Next test")
        caplog.clear()

        # test something longer
        time_logger = SamplingTimeIntervalLoggingReporter(
            logger,
            callback=hello_log,
            state_includes=state_includes,
            sampling_time_interval=(2 * openmm.unit.femtosecond),
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        simulation.reporters.append(time_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(2)
            assert len(caplog.records) == 1
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)

        assert len(caplog.records) == 5
        caplog.clear()


class Test_HeartBeatLoggingReporter:

    def test_logging_callback(self, sim_components, caplog):

        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-HeartBeatLoggingReporter"
        logger = logging.getLogger(logger_name)

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)

        reporter = HeartBeatLoggingReporter(
            logger,
            step_interval=2,
            start_time=time.time(),
        )
        with caplog.at_level(logging.INFO, logger_name):
            reporter.logging_callback(
                logger,
                simulation,
                omm_state,
            )

        assert len(caplog.records) == 1

    def test_simulation(self, sim_components, caplog):

        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-HeartBeatLoggingReporter"
        logger = logging.getLogger(logger_name)

        time_logger = HeartBeatLoggingReporter(
            logger,
            step_interval=2,
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        simulation.reporters.append(time_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(1)

        assert len(caplog.records) == 0
        caplog.clear()

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(1)

        assert len(caplog.records) == 1
        caplog.clear()

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(10)

        assert len(caplog.records) == 5
        caplog.clear()


class Test_EnergyLoggingReporter:

    def test_logging_callback(self, sim_components, caplog):

        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-EnergyLoggingReporter"
        logger = logging.getLogger(logger_name)

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        # do a step to compute energies
        simulation.context.getIntegrator().step(1)
        _omm_state = simulation.context.getState(energy=True)

        reporter = EnergyLoggingReporter(
            logger,
            sampling_time_interval=(1 * openmm.unit.femtosecond),
            start_time=time.time(),
        )
        with caplog.at_level(logging.INFO, logger_name):
            reporter.logging_callback(
                logger,
                simulation,
                _omm_state,
            )

        assert len(caplog.records) == 1

    def test_simulation(self, sim_components, caplog):
        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-EnergyLoggingReporter"
        logger = logging.getLogger(logger_name)

        energy_logger = EnergyLoggingReporter(
            logger,
            sampling_time_interval=(STEP_TIME * 2),
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        simulation.reporters.append(energy_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(2)
            assert len(caplog.records) == 1
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)

        assert len(caplog.records) == 5
        caplog.clear()


class Test_UnitCellLoggingReporter:

    def test_logging_callback(self, sim_components, caplog):

        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-UnitCellLoggingReporter"
        logger = logging.getLogger(logger_name)

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        # do a step to compute energies
        simulation.context.getIntegrator().step(1)
        _omm_state = simulation.context.getState(energy=True)

        reporter = UnitCellLoggingReporter(
            logger,
            sampling_time_interval=(1 * openmm.unit.femtosecond),
            start_time=time.time(),
        )
        with caplog.at_level(logging.INFO, logger_name):
            reporter.logging_callback(
                logger,
                simulation,
                _omm_state,
            )

        assert len(caplog.records) == 1

    def test_simulation(self, sim_components, caplog):
        omm_state, sim_args = sim_components[0], sim_components[1:]

        topology, system, integrator, platform = sim_args

        logger_name = "test-UnitCellLoggingReporter"
        logger = logging.getLogger(logger_name)

        energy_logger = UnitCellLoggingReporter(
            logger,
            sampling_time_interval=(STEP_TIME * 2),
            start_time=time.time(),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            copy.deepcopy(integrator),
            platform,
        )
        simulation.context.setState(omm_state)
        simulation.reporters.append(energy_logger)

        with caplog.at_level(logging.INFO, logger_name):
            simulation.step(2)
            assert len(caplog.records) == 1
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)
            simulation.step(2)

        assert len(caplog.records) == 5
        caplog.clear()
