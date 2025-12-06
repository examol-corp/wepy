import logging
from typing import Callable
from collections.abc import Collection
import openmm.app
import openmm
import openmm.unit
import attrs

from .reporter import (
    OpenMMReporter,
    OpenMMGetStateKeys,
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

LoggingReporterFactory = Callable[[logging.Logger], LoggingReporter]

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
    ) -> None:

        super().__init__(
            logger=logger,
            callback=callback,
            state_includes=state_includes,
        )
        self.step_interval = step_interval

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
    step_size: openmm.unit.Quantity
    sampling_time_interval: openmm.unit.Quantity

    def __init__(
        self,
        logger: logging.Logger,
        callback: LoggingReporterCallback,
        state_includes: Collection[OpenMMGetStateKeys],
        step_size: openmm.unit.Quantity,
        sampling_time_interval: openmm.unit.Quantity,
    ) -> None:

        super().__init__(logger, callback, state_includes)
        self.sampling_time_interval = sampling_time_interval
        self.step_size = step_size

    def describeNextReport(
        self, simulation: openmm.app.Simulation
    ) -> OpenMMReporterNextReport:

        _unit = openmm.unit.attosecond

        curr_sampling_time: openmm.unit.Quantity = simulation.context.getTime()

        # avg_step_time: openmm.unit.Quantity = curr_sampling_time / simulation.currentStep

        sampling_time_left = (
            self.sampling_time_interval.value_in_unit(_unit) - (
                curr_sampling_time.value_in_unit(_unit)
                % self.sampling_time_interval.value_in_unit(_unit)
            )
        ) * _unit

        if sampling_time_left < (0. * _unit):
            estimated_steps_left = 0

        else:
            estimated_steps_left = (
                round(sampling_time_left.value_in_unit(_unit)) // round(self.step_size.value_in_unit(_unit))
            )

        return OpenMMReporterNextReport(
            steps=estimated_steps_left,
            include=list(self.state_includes),
            periodic=False,
        )


class HeartBeatLoggingReporter(StepIntervalLoggingReporter):

    def __init__(
        self,
        logger: logging.Logger,
        step_interval: int
    ) -> None:

        super().__init__(
            logger=logger,
            callback=self.logging_callback,
            state_includes=[],
            step_interval=step_interval,
        )
    
    @staticmethod
    def logging_callback(
            logger: logging.Logger,
            simulation: openmm.app.Simulation,
            state: openmm.State,
    ) -> None:

        # TODO: make this adaptive to reduce zeros etc.
        sim_time = simulation.context.getTime()
        sim_time_mag = sim_time.value_in_unit(openmm.unit.picosecond)
        sim_steps = simulation.context.getStepCount()

        logger.info(f"OpenMM simulation progress: sim_time={sim_time_mag:.4f} ps, sim_steps={sim_steps}")




@attrs.define
class HeartBeatLoggingReporterFactory:

    step_interval: int

    def __call__(self, logger: logging.Logger) -> HeartBeatLoggingReporter:

        return HeartBeatLoggingReporter(
            logger=logger,
            step_interval=self.step_interval,
        )
# TODO:
# class EnergyLoggingReporter()
