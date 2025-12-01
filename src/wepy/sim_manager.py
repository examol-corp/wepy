"""Module for the main simulation management class.

All component class interfaces are set by how the manager interacts
with them.

Managers should implement a three phase protocol for running
simulations:

- init
- run_simulation
- cleanup

The separate `init` method is different than the constructor
`__init__` method and instead calls the special `init` method on all
wepy components (runner, resampler, boundary conditions, and
reporters) at runtime.

This allows for a things that need to be done at runtime before a
simulation begins, e.g. opening files, that you don't want done at
construction time.

This is useful for orchestration because the complete simulation
'image' can be made before runtime (and pickled or otherwise
persisted) without producing external effects.

This is used primarily for reporters, which perform I/O, and work
mappers which may spawn processes.

The `cleanup` method should be called either when the simulation ends
normally as well as when the simulation ends abnormally.

This allows file handles to be closed and processes to be killed at
the end of a simulation and upon failure.


The base methods for running simulations is `run_cycle` which runs
a cycle of weighted ensemble given the state of all the components.

The simulation manager should provide multiple ways of running
simulations depending on if the number of cycles is known up front or
to be determined adaptively (e.g. according to some time limit).

"""

# Standard Library
import logging
from typing import Final, Any, TypedDict, Generic, TypeVar

logger = logging.getLogger(__name__)
# Standard Library
import time
from copy import deepcopy

# First Party Library
from wepy.boundary_conditions.boundary import BoundaryConditions
from wepy.reporter.reporter import Reporter
from wepy.resampling.resamplers.resampler import Resampler
from wepy.runners.runner import Runner
from wepy.walker import Walker
from wepy.work_mapper.base import WorkMapper
from wepy.work_mapper.serial import SerialMapper
from wepy.monitor import Monitor

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
    
State_ = TypeVar("State_")
class Manager(Generic[State_]):
    """The class that coordinates wepy simulations.

    The Manager class is the lynchpin of wepy simulations and is where
    all the different components are composed.

    Strictly speaking the Manager defines the interfaces each
    component must provide to function.

    Developers can call `run_cycle` directly but the following
    convenience functions are provided to run many cycles in
    succession as a single 'run' with consecutive cycle idxs:

    - run_simulation_by_time
    - run_simulation

    The corresponding 'continue' run methods will simply pass a run
    index to reporters indicating that the run continues another.

    For these run methods the `init` method is called followed by
    iterative calls to `run_cycle` and finally with a call to
    `cleanup`.

    The order of application of wepy components are:

    - runner
    - boundary_conditions
    - resampler
    - reporters

    """

    init_walkers: list[Walker[State_]]
    n_init_walkers: int
    runner: Runner
    resampler: Resampler
    boundary_conditions: BoundaryConditions | None
    work_mapper: WorkMapper
    reporters: list[Reporter]
    monitor: Monitor | None


    REPORT_ITEM_KEYS: Final[tuple[str, ...]] = (
        "cycle_idx",
        "n_segment_steps",
        "new_walkers",
        "resampled_walkers",
        "warp_data",
        "bc_data",
        "progress_data",
        "resampling_data",
        "resampler_data",
        "worker_segment_times",
        "cycle_runner_time",
        "cycle_bc_time",
        "cycle_resampling_time",
    )
    """Keys of values that will be passed to reporters.

    This indicates the values that the reporters will have access to.
    """

    def __init__(
        self,
        init_walkers: list[Walker[State_]],
        runner: Runner,
        resampler: Resampler,
        work_mapper: WorkMapper | None = None,
        boundary_conditions: BoundaryConditions | None = None,
        reporters: list[Reporter] | None = None,
        sim_monitor: Monitor | None = None,
    ) -> None:
        """Constructor for Manager.

        Arguments:
        ---------
        init_walkers : list of walkers
            The list of the initial walkers that will be run.

        runner : object implementing the Runner interface
            The runner to be used for propagating sampling segments of walkers.

        work_mapper : object implementing the WorkMapper interface
            The object that will be used to perform a set of runner
            segments in a cycle.

        resampler : object implementing the Resampler interface
            The resampler to be used in the simulation

        boundary_conditions : object implementing BoundaryCondition interface, optional
            The boundary conditions to apply to walkers

        reporters : list of objects implenting the Reporter interface, optional
            Reporters to be used. You should provide these if you want to keep data.

        sim_monitor: Monitoring object. Can be used to report metrics
            outside of simulation data flow.

        Warnings:
        --------
        While reporters are strictly optional, you probably want to
        provide some because the simulation manager provides no
        utilities for saving data from the simulations except for the
        walkers at the end of a cycle or simulation.

        See Also:
        --------
        wepy.reporter.hdf5 : The standard reporter for molecular simulations in wepy.

        """

        self.init_walkers = init_walkers
        self.n_init_walkers = len(init_walkers)

        # the runner is the object that runs dynamics
        self.runner = runner
        # the resampler
        self.resampler = resampler
        # object for boundary conditions
        self.boundary_conditions = boundary_conditions

        # the method for writing output
        if reporters is None:
            self.reporters = []
        else:
            self.reporters = reporters

        if work_mapper is None:
            self.work_mapper = SerialMapper()
        else:
            self.work_mapper = work_mapper

        ## Monitor
        self.monitor = sim_monitor

        # used to have a record of the last report for the simulation
        # monitor without breaking the API. Ugly but I don't want to
        # break it and no one cares about this anyhow
        self._last_report: CycleReportDict | None = None


    def init(
        self,
        num_workers: int | None = None,
        continue_run: int | None = None,
    ) -> None:
        """Initialize wepy configuration components for use at runtime.

        This `init` method is different than the constructor
        `__init__` method and instead calls the special `init` method
        on all wepy components (runner, resampler, boundary
        conditions, and reporters) at runtime.

        This allows for a things that need to be done at runtime before a
        simulation begins, e.g. opening files, that you don't want done at
        construction time.

        It calls the `init` methods on:

        - work_mapper
        - reporters

        Passes the segment_func of the runner and the number of
        workers to the work_mapper.

        Passes the following things to each reporter `init` method:

        - init_walkers
        - runner
        - resampler
        - boundary_conditions
        - work_mapper
        - reporters
        - continue_run

        Parameters
        ----------
        num_workers : int
            The number of workers to use in the work mapper.
             (Default value = None)
        continue_run : int
            Index of a run this one is continuing.
             (Default value = None)

        """

        logger.info("Starting simulation")

        # initialize the monitoring object

        # TODO: do we need to supply the port here? I don't want to
        # add it to the interface... Should be pre-parametrized
        if self.monitor is not None:
            logger.info("Initializing monitoring")
            self.monitor.init()

        # initialize the work_mapper with the function it will be
        # mapping and the number of workers, this may include things like starting processes
        # etc.
        logger.info("Initializing work_mapper")
        self.work_mapper.init(
            segment_func=self.runner.run_segment,
            num_workers=num_workers,
        )

        # init the reporter
        for reporter in self.reporters:
            logger.info(f"Initializing reporter: {reporter}")
            reporter.init(
                init_walkers=self.init_walkers,
                runner=self.runner,
                resampler=self.resampler,
                boundary_conditions=self.boundary_conditions,
                work_mapper=self.work_mapper,
                reporters=self.reporters,
                continue_run=continue_run,
            )

        logger.info("Finished sim_manager initialization")

    def cleanup(self) -> None:
        """Perform cleanup actions for wepy configuration components.

        Allow components to perform actions before ending the main
        simulation manager process.

        Calls the `cleanup` method on:

        - work_mapper
        - reporters

        Passes nothing to the work mapper.

        Passes the following to each reporter:

        - runner
        - work_mapper
        - resampler
        - boundary_conditions
        - reporters

        """

        logger.info("Running cleanup")

        if self.monitor is not None:
            logger.info("Cleaning up monitoring")
            self.monitor.cleanup()

        # cleanup the mapper
        logger.info("Cleaning up work_mapper")
        self.work_mapper.cleanup()

        # cleanup things associated with the reporter
        for reporter in self.reporters:
            logger.info(f"Cleaning up reporter: {reporter}")
            reporter.cleanup(
                runner=self.runner,
                work_mapper=self.work_mapper,
                resampler=self.resampler,
                boundary_conditions=self.boundary_conditions,
                reporters=self.reporters,
            )

        logger.info("Finished cleanup")

    def run_segment(
        self,
        states: list[State_],
        segment_length: int,
        cycle_idx: int,
    ) -> list[State_]:
        """Run a time segment for all walkers using the available workers.

        Maps the work for running each segment for each walker using
        the work mapper.

        Walkers will have the same weights but different states.

        Parameters
        ----------
        walkers : list[Walker]
            List of walkers

        segment_length : int
            Number of steps to run in each segment.

        cycle_idx : int
            Cycle index

        Returns
        -------
        new_walkers : list[Walker]
           The walkers after the segment of sampling simulation.
        """

        num_walkers = len(states)

        logger.info("Starting segment")

        segment_lengths = [segment_length for i in range(num_walkers)]
        cycle_idxs = [cycle_idx for i in range(num_walkers)]
        walker_idxs = [walker_idx for walker_idx in range(num_walkers)]
        try:
            new_states = list(
                self.work_mapper.map(
                    # args, which must be supported by the map function
                    states,
                    segment_lengths,
                    # kwargs which are optionally recognized by the map function
                    cycle_idx=cycle_idxs,
                    walker_idx=walker_idxs,
                )
            )

        except Exception as exception:
            logger.info("Exception encountered in segment calculations. Cleaning up before raising.")
            # get the errors from the work mapper error queue
            self.cleanup()

            logger.info("Failure cleanup complete, reraising error.")

            # report on all of the errors that occured
            raise exception

        logger.info("Ending segment")

        return new_states

    def run_cycle(
        self,
        walkers: list[Walker[State_]],
        n_segment_steps: int,
        cycle_idx: int,
        runner_opts=None,
    ) -> tuple[
        list[Walker[State_]],
        tuple[Runner, BoundaryConditions | None, Resampler],
    ]:
        """Run a full cycle of weighted ensemble simulation using each
        component.

        The order of application of wepy components are:

        - runner
        - boundary_conditions
        - resampler
        - reporters

        The `init` method should have been called before this or
        components may fail.

        This method is not idempotent and will alter the state of wepy
        components.

        The cycle is not kept as a state variable of the simulation
        manager and so myst be provided here. This motivation for this
        is that a cycle index is really a property of a run and runs
        can be composed in many ways and is then handled by
        higher-level methods calling run_cycle.

        Each component should implement its respective interface to be
        called in this method, the order and names of the methods
        called are as follows:

        1. runner.pre_cycle
        2. run_segment -> work_mapper.map(runner.run_segment)
        3. runner.post_cycle
        4. boundary_conditions.warp_walkers (if present)
        5. resampler.resample
        6. reporter.report for all reporters

        The pre and post cycle calls to the runner allow for a parity
        of one call per cycle to the runner.

        The boundary_conditions component is optional, as are the
        reporters (although it won't be very useful to run this
        without any).

        Parameters
        ----------
        walkers : list of walkers

        n_segment_steps : int
            Number of steps to run in each segment.

        cycle_idx : int
            The index of this cycle.

        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            The runner, resampler, and boundary conditions
            objects at the end of the cycle.

        See Also
        --------
        run_simulation : To run a simulation by the number of cycles
        run_simulation_by_time

        """

        logger.info("Running simulation cycle")

        if runner_opts is None:
            runner_opts = {}

        # run the runner pre-cycle hook
        start = time.time()

        logger.info("Running Runner.pre_cycle hook")
        self.runner.pre_cycle(
            **runner_opts,
        )

        end = time.time()
        runner_precycle_time = end - start
        logger.info(f"Precycle time: {runner_precycle_time}")

        # run the segment
        start = time.time()

        logger.info("Running state propagation segment")
        new_states = self.run_segment(
            [walker.state for walker in walkers],
            n_segment_steps,
            cycle_idx,
        )
        logger.info("Finished state propagation segment")

        new_walkers = [
            Walker(
                state=new_state,
                weight=walker.weight
            )
            for walker, new_state
            in zip(walkers, new_states, strict=True)
        ]

        end = time.time()
        sim_manager_segment_time = end - start
        logger.info(f"Segment duration: {sim_manager_segment_time}")

        runner_splits = self.runner.get_last_cycle_segments_split_times()

        logger.info("Running Runner.post_cycle hook")
        # run post-cycle hook
        start = time.time()

        self.runner.post_cycle()

        end = time.time()
        runner_postcycle_time = end - start

        logger.info(f"Post cycle duration: {runner_postcycle_time}")

        # boundary conditions should be optional;

        # initialize the warped walkers to the new_walkers and
        # change them later if need be
        warped_walkers = new_walkers
        warp_data = []
        bc_data = []
        progress_data = {}
        bc_time = 0.0
        if self.boundary_conditions is not None:
            logger.info("Boundary conditions were provided, applying.")
            # apply rules of boundary conditions and warp walkers through space
            start = time.time()
            logger.info("Starting boundary conditions calculations")
            bc_results = self.boundary_conditions.warp_walkers(new_walkers, cycle_idx)
            end = time.time()
            bc_time = end - start

            # warping results
            warped_walkers = bc_results[0]
            warp_data = bc_results[1]
            bc_data = bc_results[2]
            progress_data = bc_results[3]

            logger.info(f"Boundary condition duration: {bc_time}")

            if len(warp_data) > 0:
                logger.info(f"Returned warp record in cycle {cycle_idx}")

        # resample walkers
        logger.info("Starting resampler phase.")
        start = time.time()

        resampling_results = self.resampler.resample(warped_walkers)

        end = time.time()
        resampling_time = end - start

        logger.info(f"Resampling duration: {resampling_time}")

        resampled_walkers = resampling_results[0]
        resampling_data = resampling_results[1]
        resampler_data = resampling_results[2]

        # make a dictionary of all the results that will be reported
        seg_times = {}
        sampling_time = None

        if (seg_times := self.work_mapper.get_worker_segment_times()) is not None:
            logger.info("Segment timings provided by work mapper, recording.")

            # count up the total sampling time from the segments
            sampling_time = 0.
            for (
                worker_id,
                segments_times,
            ) in seg_times.items():
                for seg_time in segments_times:
                    sampling_time += seg_time

            # calculate the overhead for logging
            sim_manager_segment_overhead_time = sim_manager_segment_time - sampling_time

            logger.info(f"Simulation manager overhead time: {sim_manager_segment_overhead_time}")


        else:
            logger.info("Worker segment times not provided")
            sim_manager_segment_overhead_time = 0.0

        report = CycleReportDict({
            "cycle_idx": cycle_idx,
            "new_walkers": new_walkers,
            "warp_data": warp_data,
            "bc_data": bc_data,
            "progress_data": progress_data,
            "resampling_data": resampling_data,
            "resampler_data": resampler_data,
            "n_segment_steps": n_segment_steps,
            "resampled_walkers": resampled_walkers,
            # timings
            "runner_precycle_time": runner_precycle_time,
            "runner_postcycle_time": runner_postcycle_time,
            "sim_manager_segment_overhead_time": sim_manager_segment_overhead_time,
            "runner_splits_time": runner_splits,
            "worker_segment_times": seg_times,
            "cycle_sim_manager_segment_time": sim_manager_segment_time,
            "cycle_runner_time": sim_manager_segment_time,
            "cycle_bc_time": bc_time,
            "cycle_resampling_time": resampling_time,
        })

        self._last_report = report

        logger.info("Starting reporting")
        # report results to the reporters
        for reporter in self.reporters:
            logger.info(f"Reporting with reporter: {reporter}")
            reporter.report(**report)

        # run the simulation monitor to get metrics on everything
        if self.monitor is not None:
            logger.info("Running cycle monitoring")
            self.monitor.cycle_monitor(self, resampled_walkers)

        logger.info("Done: returning walkers")
        return resampled_walkers, (self.runner, self.boundary_conditions, self.resampler)

    def run_simulation_by_time(
            self,
            run_time: float,
            segments_length: int,
            num_workers: int | None = None,
    ) -> tuple[
        list[Walker[State_]],
        tuple[Runner, BoundaryConditions | None, Resampler],
    ]:
        """Run a simulation for a certain amount of time.

        This starts timing as soon as this is called. If the time
        before running a new cycle is greater than the runtime the run
        will exit after cleaning up. Once a cycle is started it may
        also run over the wall time.

        All this does is provide a run idx to the reporters, which is
        the run that is intended to be continued. This simulation
        manager knows no details and is left up to the reporters to
        handle this appropriately.

        Parameters
        ----------
        run_time : float
            The time to run in seconds.

        segments_length : int
            The number of steps for each runner segment.

        num_workers : int
            The number of workers to use for the work mapper.
             (Default value = None)

        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            Deep copies of the runner, resampler, and boundary
            conditions objects at the end of the cycle.


        """
        start_time = time.time()
        self.init(num_workers=num_workers)
        cycle_idx = 0
        walkers = self.init_walkers
        while time.time() - start_time < run_time:
            logger.info(
                "starting cycle {} at time {}".format(
                    cycle_idx, time.time() - start_time
                )
            )

            walkers, filters = self.run_cycle(walkers, segments_length, cycle_idx)

            logger.info(
                "ending cycle {} at time {}".format(cycle_idx, time.time() - start_time)
            )

            cycle_idx += 1

        logger.info("Cleaning up simulation")
        self.cleanup()

        return walkers, deepcopy(filters)

    def run_simulation(
        self,
        n_cycles: int,
        segment_lengths: int,
        num_workers: int | None = None,
        continue_run_idx: int | None = None,
    ) -> tuple[
        list[Walker[State_]],
        tuple[Runner, BoundaryConditions, Resampler],
    ]:
        """Run a simulation for an explicit number of cycles.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to perform.

        segment_lengths : int
            The number of steps for each runner segment.

        num_workers : int
            The number of workers to use for the work mapper.
             (Default value = None)

        continue_run_idx: Index of the run you are continuing, optional.


        Returns
        -------
        new_walkers : list of walkers
            The resulting walkers of the cycle

        sim_components : list
            Deep copies of the runner, boundary conditions, and
            resampler objects at the end of the simulation.

        """

        logger.info("Running simulation init hook")
        self.init(num_workers=num_workers, continue_run=continue_run_idx)

        if type(segment_lengths) == int:
            logger.info("Single number of steps provided for simulation, using this for all cycles.")
            segment_lengths = [segment_lengths for _ in range(n_cycles)]

        walkers = self.init_walkers

        logger.info("Starting main simulation loop over cycles")
        # the main cycle loop
        for cycle_idx in range(n_cycles):
            logger.info(f"Running cycle: {cycle_idx}")
            walkers, filters = self.run_cycle(
                walkers, segment_lengths[cycle_idx], cycle_idx,
            )
            logger.info(f"Finished running cycle: {cycle_idx}")

            # run the simulation monitor to get metrics on everything
            if self.monitor is not None:
                logger.info("Running monitoring cycle_monitor hook")
                self.monitor.cycle_monitor(self, walkers)

        logger.info("Running simulation cleanup")
        self.cleanup()
        logger.info("Simulation cleanup complete")

        return walkers, deepcopy(tuple(filters))

    def run_simulation_by_time(
        self,
        run_time: int,
        segments_length: int,
        num_workers: int | None = None,
        continue_run_idx: int | None = None,
    ) -> tuple[
        list[Walker[State_]],
        tuple[Runner, BoundaryConditions | None, Resampler],
    ]:
        """Continue a simulation with a separate run by time.

        This starts timing as soon as this is called. If the time
        before running a new cycle is greater than the runtime the run
        will exit after cleaning up. Once a cycle is started it may
        also run over the wall time.

        All this does is provide a run idx to the reporters, which is
        the run that is intended to be continued. This simulation
        manager knows no details and is left up to the reporters to
        handle this appropriately.

        """

        start_time = time.time()
        logger.info(f"Simulation start time: {start_time}")

        logger.info("Running simulation init hook")
        self.init(num_workers=num_workers, continue_run=continue_run_idx)

        cycle_idx = 0
        walkers = self.init_walkers
        while time.time() - start_time < run_time:
            logger.info(
                "starting cycle {} at time {}".format(
                    cycle_idx, time.time() - start_time
                )
            )

            walkers, filters = self.run_cycle(walkers, segments_length, cycle_idx)

            logger.info(
                "ending cycle {} at time {}".format(cycle_idx, time.time() - start_time)
            )

            # run the simulation monitor to get metrics on everything
            if self.monitor is not None:
                logger.info("Running cycle_monitor hook")
                self.monitor.cycle_monitor(self, walkers)

            cycle_idx += 1

        logger.info("Running simulation cleanup")
        self.cleanup()
        logger.info("Simulation cleanup complete")

        return walkers, filters
