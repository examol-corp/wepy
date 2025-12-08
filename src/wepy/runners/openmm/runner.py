# Standard Library
from typing import Any, Annotated, TypedDict, NotRequired, Literal, Final, Self, get_args, TypeAlias, Callable
import logging
import multiprocessing as mp
import itertools
import time
from warnings import warn
import copy

# Third Party Library
from immutables import Map as frozenmap
import attrs
import numpy as np

from .reporter import OpenMMReporter

logger = logging.getLogger(__name__)

try:
    import mdtraj
except ModuleNotFoundError:
    warn("Module 'mdtraj' not found, those features will not be available.")

try:
    # Third Party Library
    import openmm
    import openmm.app
    import openmm.unit
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "OpenMM has not been installed, which this runner requires."
    )

# First Party Library
from wepy.runners.runner import Runner, RunnerStatus, RunSegmentData, RunnerStateMachine, RunnerEvent, RunnerStateError
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.walker import WalkerState
from .state import OpenMMState, OpenMMStateWrapper, get_context_state
from .logger import HeartBeatLoggingReporterFactory, LoggingReporterFactory

PlatformKwargs = dict[str, str]

OpenMMPlatformName = Literal["Reference", "CPU", "CUDA", "OpenCL", "HIP"]
GPU_PLATFORMS = frozenset({"CUDA", "OpenCL", "HIP"})



GET_STATE_DEFAULT_KEYS = frozenset({
            "positions",
            "velocities",
            "forces",
            "parameters",
            "parameter_derivatives",
            "kinetic_energy",
            "potential_energy",
            "time",
            "box_vectors",
            "box_volume",
})

DEFAULT_OPENMM_REPORTER_FACTORIES = [
    # default heart beat every 500 steps
    HeartBeatLoggingReporterFactory(step_interval=500),
]

@attrs.define
class OpenMMRunnerSegmentSplitTime:
    gen_sim_time: float
    steps_time: float
    get_state_time: float

@attrs.define
class OpenMMRunnerSegmentData(RunSegmentData):
    segment_split_time: float
    openmm_segment_split_time: OpenMMRunnerSegmentSplitTime

# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner):
    """Runner for OpenMM simulations."""

    system: openmm.System
    topology: openmm.app.Topology
    integrator: openmm.Integrator
    platform_name: str | None
    global_platform_kwargs: PlatformKwargs | None
    enforce_box: bool
    get_state_keys: frozenset[str]
    openmm_reporter_factories: list[LoggingReporterFactory] | None

    state_machine: RunnerStateMachine

    _openmm_reporters: list[OpenMMReporter] | None
    _init_time: int | None
    _pre_cycle_time: int | None

    def __init__(
        self,
        system: openmm.System,
        topology: openmm.app.Topology,
        integrator: openmm.Integrator,
        platform_name: str | None = None,
        global_platform_kwargs: PlatformKwargs | None = None,
        enforce_box: bool = False,
        get_state_keys: frozenset[str] = GET_STATE_DEFAULT_KEYS,
        openmm_reporter_factories: list[LoggingReporterFactory] | None = None,
    ) -> None:

        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.platform_name = platform_name
        self.global_platform_args = global_platform_kwargs
        self.enforce_box = enforce_box
        self.get_state_keys = get_state_keys

        logger.warning("No OpenMM reporter factories configured.")
        self.openmm_reporter_factories = openmm_reporter_factories if openmm_reporter_factories is not None else []

        self._openmm_reporters = None
        self._init_time = None
        self._pre_cycle_time = None

        self.state_machine = RunnerStateMachine()

    @property
    def status(self) -> RunnerStatus:
        return self.state_machine.state

    def init(self) -> None:

        self.state_machine.validate_event(RunnerEvent.INIT)

        self._init_time = time.time()
        logger.info(f"Initialized runner at time: {self._init_time} s")

        self.state_machine.send(RunnerEvent.INIT)

    def pre_cycle(
        self,
    ) -> None:

        self.state_machine.validate_event(RunnerEvent.PRE_CYCLE)

        self._pre_cycle_time = time.time()
        logger.info(f"Runner pre_cycle time: {self._pre_cycle_time} s")

        self.state_machine.send(RunnerEvent.PRE_CYCLE)

    
    def run_segment(
        self,
        walker_state: OpenMMState,
        segment_length: int,
        platform_name: OpenMMPlatformName | None = None,
        platform_kwargs: PlatformKwargs | None = None,
    ) -> tuple[
        OpenMMState,
        OpenMMRunnerSegmentData,
    ]:
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : The walker for which dynamics will be propagated.

        segment_length : The numerical value that specifies how much dynamics are to be run.

        platform_kwargs : Key-values to set for a platform with
            platform.setPropertyDefaultValue for this segment only.


        Returns
        -------
        new_walker_state : Walker after dynamics was run, only the state should be modified.

        """

        if self.status != RunnerStatus.PRE_CYCLE:
            raise RunnerStateError(
                f"Cannot run a segment in state ({self.status.name}:{self.status.value})"
            )

        logger.info("Running OpenMM MD segment")

        run_segment_start = time.time()

        # set the kwargs that will be passed to getState
        gen_sim_start = time.time()

        # TODO: refactor this as an integrator spec as the object
        # attribute to avoid needing to do this and make this
        # interface explicit

        # make a copy of the integrator for this particular segment,
        # otherwise the object attribute will get bound to the context
        new_integrator = copy.copy(self.integrator)
        # force setting of random seed to 0, which is a special
        # value that forces the integrator to choose another
        # random number
        logger.info("Setting random seed to special value: 0")
        new_integrator.setRandomNumberSeed(0)

        ## Platform

        logger.info(f"'platform_kwargs' passed to 'run_segment' : {platform_kwargs}")

        # create simulation object

        ## create the platform and customize

        # if a platform was given we use it to make a Simulation object
        if platform_name is not None:
            logger.info("Using platform configured in code.")

            # get the platform by its name to use
            platform = openmm.Platform.getPlatformByName(platform_name)
            logger.info(f"Platform object created: {platform}")

            # set properties from the kwargs if they apply to the platform
            for key, value in platform_kwargs.items():
                if key in platform.getPropertyNames():
                    logger.info(f"Setting platform property: {key} : {value}")
                    platform.setPropertyDefaultValue(key, value)

                else:
                    logger.warning(
                        f"Platform kwargs given ({key} : {value}) "
                        f"but is not valid for this platform ({platform_name})"
                    )

            # make a new simulation object
            logger.info("Construction Simulation and context")
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator, platform
            )

        # otherwise just use the default or environmentally defined one
        else:
            logger.info("Using OpenMM default platform resolution.")
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator
            )

        # Generate new reporters for each segment so they don't step
        # on each other's state
        logger.info("Generating OpenMM reporters for this segment.")
        openmm_reporters = []
        for omm_reporter_factory in self.openmm_reporter_factories:
            logger.info(f"Generating and configuring reporter for factory: {omm_reporter_factory}")
            openmm_reporters.append(
                omm_reporter_factory(
                    logger,
                    start_time=run_segment_start,
                )
            )

        logger.info("Registering OpenMM Simulation reporters")
        simulation.reporters = openmm_reporters

        # generate a sim state
        logger.info("Generating openmm.State from input OpenMMState")
        state_wrapper = walker_state.to_state_wrapper()


        # set in the context
        logger.info("Setting openmm.State into current context")
        simulation.context.setState(state_wrapper.state)

        gen_sim_end = time.time()
        gen_sim_time = gen_sim_end - gen_sim_start

        logger.info("Time to generate the system: {}".format(gen_sim_time))

        # actually run the simulation

        steps_start = time.time()

        # Run the simulation segment for the number of time steps
        logger.info("Running MD steps")
        simulation.step(segment_length)

        steps_end = time.time()
        steps_time = steps_end - steps_start

        logger.info(f"Time to run {segment_length} sim steps: {steps_time} s")

        get_state_start = time.time()

        # generate the new state

        logger.info(f"Fetching fields {self.get_state_keys} from context state")

        new_omm_state = get_context_state(
            simulation.context,
            self.get_state_keys,
        )

        new_state_wrapper = OpenMMStateWrapper(new_omm_state)
        new_state = OpenMMState.from_state_wrapper(new_state_wrapper)
        
        get_state_end = time.time()
        get_state_time = get_state_end - get_state_start
        logger.info("Getting context state time: {}".format(get_state_time))

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logger.info("Total internal run_segment time: {}".format(run_segment_time))

        segment_data = OpenMMRunnerSegmentData(
            segment_split_time=run_segment_time,
            openmm_segment_split_time=OpenMMRunnerSegmentSplitTime(
                gen_sim_time=gen_sim_time,
                steps_time=steps_time,
                get_state_time=get_state_time,
            ),
        )

        return new_state, segment_data

    def post_cycle(self, segments_data: list[OpenMMRunnerSegmentData]) -> None:

        self.state_machine.send(RunnerEvent.POST_SEGMENT)
        logger.info("Nothing to do")
        self.state_machine.send(RunnerEvent.POST_CYCLE)

@attrs.define
class OpenMMRunnerFactory:

    system: openmm.System
    topology: openmm.app.Topology
    integrator: openmm.Integrator
    platform_name: str | None = None
    global_platform_kwargs: PlatformKwargs | None = None
    enforce_box: bool = False
    get_state_keys: frozenset[str] = attrs.field(default=GET_STATE_DEFAULT_KEYS)
    openmm_reporter_factories: list[LoggingReporterFactory] | None = attrs.field(default=DEFAULT_OPENMM_REPORTER_FACTORIES)

    def __call__(self) -> OpenMMRunner:

        return OpenMMRunner(
            system=copy.deepcopy(self.system),
            topology=copy.deepcopy(self.topology),
            integrator=copy.deepcopy(self.integrator),
            platform_name=self.platform_name,
            global_platform_kwargs=self.global_platform_kwargs,
            enforce_box=self.enforce_box,
            get_state_keys=self.get_state_keys,
            openmm_reporter_factories=self.openmm_reporter_factories,
        )


# class OpenMMCPUWorker(Worker):
#     """Worker for OpenMM GPU simulations (CUDA or OpenCL platforms).

#     This is intended to be used with the wepy.work_mapper.WorkerMapper
#     work mapper class.

#     This class must be used in order to ensure OpenMM runs jobs on the
#     appropriate GPU device.

#     """

#     NAME_TEMPLATE = "OpenMMCPUWorker-{}"
#     """The name template the worker processes are named to substituting in
#     the process number."""

#     DEFAULT_NUM_THREADS = 1

#     def __init__(self, *args, **kwargs):
#         if "num_threads" not in kwargs:
#             num_threads = self.DEFAULT_NUM_THREADS
#         else:
#             num_threads = kwargs.pop("num_threads")

#         super().__init__(*args, num_threads=num_threads, **kwargs)

#     def run_task(self, task):
#         # documented in superclass

#         # make the platform kwargs dictionary
#         platform_options = {"Threads": str(self.attributes["num_threads"])}

#         # run the task and pass in the DeviceIndex for OpenMM to
#         # assign work to the correct GPU
#         return task(platform_kwargs=platform_options)


# class OpenMMGPUWorker(Worker):
#     """Worker for OpenMM GPU simulations (CUDA or OpenCL platforms).

#     This is intended to be used with the wepy.work_mapper.WorkerMapper
#     work mapper class.

#     This class must be used in order to ensure OpenMM runs jobs on the
#     appropriate GPU device.

#     """

#     NAME_TEMPLATE = "OpenMMGPUWorker-{}"
#     """The name template the worker processes are named to substituting in
#     the process number."""

#     def run_task(self, task):
#         # get the platform
#         platform = self.mapper_attributes["platform"]

#         # get the device index from the attributes
#         device_id = self.mapper_attributes["device_ids"][self._worker_idx]

#         # make the platform kwargs dictionary
#         platform_options = {"DeviceIndex": str(device_id)}

#         logger.info(f"platform={platform}, platform_options={platform_options}")

#         return task(
#             platform=platform,
#             platform_kwargs=platform_options,
#         )


# class OpenMMCPUWalkerTaskProcess(WalkerTaskProcess):
#     NAME_TEMPLATE = "OpenMM_CPU_Walker_Task-{}"

#     def run_task(self, task):
#         print("CPU Walker Task ---->", self.mapper_attributes, task, task.func)
#         if "num_threads" in self.mapper_attributes:
#             num_threads = self.mapper_attributes["num_threads"]

#             # make the platform kwargs dictionary
#             platform_options = {"Threads": str(num_threads)}

#             logger.info(f"Threads={num_threads}")

#         else:
#             platform_options = {}

#         return task(
#             platform_kwargs=platform_options,
#         )


# class OpenMMGPUWalkerTaskProcess(WalkerTaskProcess):
#     NAME_TEMPLATE = "OpenMM_GPU_Walker_Task-{}"

#     def run_task(self, task):
#         logger.info(f"Starting to run a task as worker {self._worker_idx}")

#         logger.info(f"GPU Walker Task ----> {self.mapper_attributes}")
#         # get the platform
#         platform = self.mapper_attributes["platform"]

#         # get the device index from the attributes
#         device_id = self.mapper_attributes["device_ids"][self._worker_idx]

#         # make the platform kwargs dictionary
#         platform_options = {"DeviceIndex": str(device_id)}

#         logger.info(f"platform={platform}, platform_options={platform_options}")

#         return task(
#             platform=platform,
#             platform_kwargs=platform_options,
#         )

