# Standard Library
import logging
from typing import Any, Protocol, TypedDict

# First Party Library
from wepy.boundary_conditions.boundary import BoundaryConditions
from wepy.resampling.resamplers.resampler import Resampler
from wepy.runners.runner import Runner
from wepy.walker import Walker
from wepy.work_mapper.base import WorkMapper

logger = logging.getLogger(__name__)


class ReporterError(Exception):
    pass


class SimComponentArgs(TypedDict):
    init_walkers: list[Walker]
    runner: Runner
    resampler: Resampler
    boundary_conditions: BoundaryConditions | None
    work_mapper: WorkMapper
    reporters: list["Reporter"]
    continue_run: int | None


class CycleReportDict(TypedDict):
    cycle_idx: int
    new_walkers: list[Walker]
    # TODO: types for all the Anys
    warp_data: list[Any]
    bc_data: list[Any]
    progress_data: dict[Any]
    resampling_data: Any
    resampler_data: Any
    n_segment_steps: int
    resampled_walkers: list[Walker]
    runner_precycle_time: float
    runner_postcycle_time: float
    sim_manager_segment_overhead_time: float
    runner_splits_time: dict[str, float] | None
    worker_segment_times: dict[int, list[float]] | None
    cycle_sim_manager_segment_time: float
    cycle_runner_time: float
    cycle_bc_time: float
    cycle_resampling_time: float


class Reporter(Protocol):
    """Abstract base class for wepy reporters.

    All reporters must customize and override minimally the 'report'
    method. Optionally the 'init' and 'cleanup' can be overriden.

    """

    def init(
        self,
        **kwargs: SimComponentArgs,
    ) -> None:
        """Initialization routines for the reporter at simulation runtime.

        Initialize I/O connections including file descriptors,
        database connections, timers, stdout/stderr etc.

        Void method for reporter base class.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager in this
        call.


        Parameters
        ----------
        init_walkers : list of Walker objects
            The initial walkers for the simulation.

        runner : Runner object
            The runner that will be used in the simulation.

        resampler : Resampler object
            The resampler that will be used in the simulation.

        boundary_conditions : BoundaryConditions object
            The boundary conditions taht will be used in the simulation.

        work_mapper : WorkMapper object
            The work mapper that will be used in the simulation.

        reporters : list of Reporter objects
            The list of reporters that are in the simulation.

        continue_run : int or None
            The index of the run that is being continued within this
            same file.

        """
        ...

    def report(
        self,
        **kwargs: CycleReportDict,
    ) -> None:
        """Given data concerning the main simulation components state, perform
        I/O operations to persist that data.

        Void method for reporter base class.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager.

        """
        ...

    def cleanup(
        self,
        **kwargs: SimComponentArgs,
    ) -> None:
        """Teardown routines for the reporter at the end of the simulation.

        Use to cleanly and safely close I/O connections or other
        cleanup I/O.

        Use to close file descriptors, database connections etc.

        Reporters can expect to have the following key word arguments
        passed to them during a simulation by the sim_manager.

        Parameters
        ----------
        runner : Runner object
            The runner at the end of the simulation

        work_mapper : WorkeMapper object
            The work mapper at the end of the simulation

        resampler : Resampler object
            The resampler at the end of the simulation

        boundary_conditions : BoundaryConditions object
            The boundary conditions at the end of the simulation

        reporters : list of Reporter objects
            The list of reporters at the end of the simulation

        """
        ...
