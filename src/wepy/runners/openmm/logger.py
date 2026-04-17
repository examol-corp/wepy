# Standard Library
import logging
import time
from collections.abc import Collection
from typing import Callable

# Third Party Library
import attrs
import openmm
import openmm.app
import openmm.unit

# First Party Library
from wepy.util.openmm import format_box_vectors_line

# Local Modules
from .reporter import (
    OpenMMGetStateKeys,
    OpenMMReporter,
    OpenMMReporterNextReport,
)

LoggingReporterCallback = Callable[
    [
        logging.Logger,
        openmm.app.Simulation,
        openmm.State,
    ],
    None,
]


class LoggingReporter(OpenMMReporter):
    logger: logging.Logger
    callback: LoggingReporterCallback
    state_includes: list[OpenMMGetStateKeys]

    def __init__(
        self,
        logger: logging.Logger,
        callback: LoggingReporterCallback,
        state_includes: Collection[OpenMMGetStateKeys],
    ) -> None:

        self.logger = logger
        self.callback = callback
        self.state_includes = list(state_includes)

    def report(
        self,
        simulation: openmm.app.Simulation,
        state: openmm.State,
    ) -> None:

        self.callback(
            self.logger,
            simulation,
            state,
        )


LoggingReporterFactory = Callable[
    [
        logging.Logger,
        # start_time
        int,
    ],
    LoggingReporter,
]


class StepIntervalLoggingReporter(LoggingReporter):
    """Reporter that reports at intervals in steps."""

    logger: logging.Logger
    callback: LoggingReporterCallback
    state_includes: list[OpenMMGetStateKeys]
    step_interval: int

    def __init__(
        self,
        logger: logging.Logger,
        callback: LoggingReporterCallback,
        state_includes: Collection[OpenMMGetStateKeys],
        step_interval: int,
        start_time: int,
    ) -> None:

        super().__init__(
            logger=logger,
            callback=callback,
            state_includes=state_includes,
        )
        self.step_interval = step_interval
        self.start_time = start_time

    def describeNextReport(
        self, simulation: openmm.app.Simulation
    ) -> OpenMMReporterNextReport:

        steps_left = self.step_interval - simulation.currentStep % self.step_interval

        return OpenMMReporterNextReport(
            steps=steps_left,
            include=list(self.state_includes),
            periodic=False,
        )


class SamplingTimeIntervalLoggingReporter(LoggingReporter):
    """Reporter that reports at intervals in sampling time.

    Does not work for variable step integrators currently.

    """

    logger: logging.Logger
    callback: LoggingReporterCallback
    state_includes: list[OpenMMGetStateKeys]
    sampling_time_interval: openmm.unit.Quantity

    def __init__(
        self,
        logger: logging.Logger,
        callback: LoggingReporterCallback,
        state_includes: Collection[OpenMMGetStateKeys],
        sampling_time_interval: openmm.unit.Quantity,
        start_time: int,
    ) -> None:

        super().__init__(logger, callback, state_includes)
        self.sampling_time_interval = sampling_time_interval
        self.start_time = start_time

    def describeNextReport(
        self,
        simulation: openmm.app.Simulation,
    ) -> OpenMMReporterNextReport:

        _unit = openmm.unit.attosecond

        curr_sampling_time: openmm.unit.Quantity = simulation.context.getTime()

        # get the step size from the integrator
        step_size = simulation.context.getIntegrator().getStepSize()

        sampling_time_left = (
            self.sampling_time_interval.value_in_unit(_unit)
            - (
                curr_sampling_time.value_in_unit(_unit)
                % self.sampling_time_interval.value_in_unit(_unit)
            )
        ) * _unit

        if sampling_time_left < (0.0 * _unit):
            estimated_steps_left = 0

        else:
            estimated_steps_left = round(
                sampling_time_left.value_in_unit(_unit)
            ) // round(step_size.value_in_unit(_unit))

        return OpenMMReporterNextReport(
            steps=estimated_steps_left,
            include=list(self.state_includes),
            periodic=False,
        )


class HeartBeatLoggingReporter(StepIntervalLoggingReporter):

    def __init__(
        self,
        logger: logging.Logger,
        step_interval: int,
        start_time: int,
    ) -> None:

        super().__init__(
            logger=logger,
            callback=self.logging_callback,
            state_includes=[],
            step_interval=step_interval,
            start_time=start_time,
        )

    def logging_callback(
        self,
        logger: logging.Logger,
        simulation: openmm.app.Simulation,
        state: openmm.State,
    ) -> None:

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # TODO: make this adaptive to reduce zeros etc. Currently just
        # padded to the standard 1-2 fs step time shown in picoseconds
        sim_time = simulation.context.getTime()
        sim_time_mag = sim_time.value_in_unit(openmm.unit.picosecond)
        sim_steps = simulation.context.getStepCount()

        logger.info(
            f"OpenMM simulation progress: clock_time={current_time:.4f} s, elapsed_time={elapsed_time:.4f} s, sim_time={sim_time_mag:.4f} ps, sim_steps={sim_steps}",
        )


@attrs.define
class HeartBeatLoggingReporterFactory:

    step_interval: int

    def __call__(
        self,
        logger: logging.Logger,
        start_time: int,
    ) -> HeartBeatLoggingReporter:

        return HeartBeatLoggingReporter(
            logger=logger,
            step_interval=self.step_interval,
            start_time=start_time,
        )


class EnergyLoggingReporter(SamplingTimeIntervalLoggingReporter):

    def __init__(
        self,
        logger: logging.Logger,
        sampling_time_interval: openmm.unit.Quantity,
        start_time: int,
    ) -> None:

        super().__init__(
            logger=logger,
            callback=self.logging_callback,
            state_includes=["energy"],
            sampling_time_interval=sampling_time_interval,
            start_time=start_time,
        )

    def logging_callback(
        self,
        logger: logging.Logger,
        simulation: openmm.app.Simulation,
        state: openmm.State,
    ) -> None:

        sim_time = simulation.context.getTime()
        sim_time_mag = sim_time.value_in_unit(openmm.unit.picosecond)
        sim_steps = simulation.context.getStepCount()

        _pot_e = state.getPotentialEnergy()
        _kin_e = state.getKineticEnergy()
        _tot_e = _pot_e + _kin_e

        logger.info(
            f"OpenMM simulation energy (steps={sim_steps}, sim_time={sim_time_mag:.4f} ps): "
            f"kinetic={_kin_e}, potential={_pot_e}, total={_tot_e}"
        )


@attrs.define
class EnergyLoggingReporterFactory:

    sampling_time_interval: int

    def __call__(
        self,
        logger: logging.Logger,
        start_time: int,
    ) -> EnergyLoggingReporter:

        return EnergyLoggingReporter(
            logger=logger,
            sampling_time_interval=self.sampling_time_interval,
            start_time=start_time,
        )


class UnitCellLoggingReporter(SamplingTimeIntervalLoggingReporter):

    def __init__(
        self,
        logger: logging.Logger,
        sampling_time_interval: openmm.unit.Quantity,
        start_time: int,
    ) -> None:

        super().__init__(
            logger=logger,
            callback=self.logging_callback,
            state_includes=[],
            sampling_time_interval=sampling_time_interval,
            start_time=start_time,
        )

    def logging_callback(
        self,
        logger: logging.Logger,
        simulation: openmm.app.Simulation,
        state: openmm.State,
    ) -> None:

        sim_time = simulation.context.getTime()
        sim_time_mag = sim_time.value_in_unit(openmm.unit.picosecond)
        sim_steps = simulation.context.getStepCount()

        _box_volume = state.getPeriodicBoxVolume()
        _bvs = state.getPeriodicBoxVectors()
        _bvs_line = format_box_vectors_line(_bvs)

        logger.info(
            f"OpenMM simulation unitcell (steps={sim_steps}, sim_time={sim_time_mag:.4f} ps): "
            f"volume={_box_volume}, vectors={_bvs_line}"
        )


@attrs.define
class UnitCellLoggingReporterFactory:

    sampling_time_interval: int

    def __call__(
        self,
        logger: logging.Logger,
        start_time: int,
    ) -> EnergyLoggingReporter:

        return UnitCellLoggingReporter(
            logger=logger,
            sampling_time_interval=self.sampling_time_interval,
            start_time=start_time,
        )
