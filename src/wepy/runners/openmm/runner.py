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
from wepy.util.openmm import triclinic_volume_vec3_quantity, format_box_vectors_line
from wepy.walker import WalkerState
from .state import OpenMMState, OpenMMStateWrapper, get_context_state
from .logger import HeartBeatLoggingReporterFactory, LoggingReporterFactory, EnergyLoggingReporterFactory, UnitCellLoggingReporterFactory

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

# default heart beat every 500 steps, should be around 0.5 - 1 picoseconds
_DEFAULT_HEARTBEAT_INTERVAL = 500
_DEFAULT_STATE_TIME_INTERVAL = (10 * openmm.unit.picosecond)

DEFAULT_OPENMM_REPORTER_FACTORIES = [
    HeartBeatLoggingReporterFactory(step_interval=_DEFAULT_HEARTBEAT_INTERVAL),
    UnitCellLoggingReporterFactory(sampling_time_interval=_DEFAULT_STATE_TIME_INTERVAL),
    EnergyLoggingReporterFactory(sampling_time_interval=_DEFAULT_STATE_TIME_INTERVAL),
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

def _report_simulation(simulation: openmm.app.Simulation) -> tuple[
        openmm.unit.Quantity,
        tuple[openmm.unit.Quantity, openmm.unit.Quantity, openmm.unit.Quantity],
        openmm.unit.Quantity,
        openmm.unit.Quantity,
        openmm.unit.Quantity,
]:

    # log some info on the constructed simulation
    _step_count = simulation.context.getStepCount()
    _time = simulation.context.getTime()
    _num_molecules = len(simulation.context.getMolecules())
    _platform = simulation.context.getPlatform()
    _platform_name = _platform.getName()
    _platform_prop_names = _platform.getPropertyNames()
    _props = {
        name: _platform.getPropertyValue(
            simulation.context,
            name,
        )
        for name in _platform_prop_names
    }
    _openmm_version = _platform.getOpenMMVersion()
    logger.info(f"Simulation Context: current_step={_step_count}, sampling_time={_time}, num_molecules={_num_molecules}")
    logger.info(f"Simulation Context Platform: openmm_version={_openmm_version}, name={_platform_name}, properties={_props}")

    # report on the initial state as well
    _init_state = simulation.context.getState(
        positions=False,
        velocities=True,
        forces=False,
        energy=True,
    )
    _velocities = _init_state.getVelocities()
    _vel0 = _velocities[0]
    _vel0_mag = _vel0.value_in_unit(_vel0.unit)
    _vels_zeroed = (
        np.isclose(_vel0_mag[0], 0.) and np.isclose(_vel0_mag[1], 0.) and np.isclose(_vel0_mag[2], 0.)
    )

    if _vels_zeroed:
        logger.info("Context state velocities are zeroed.")
    else:
        logger.info("Context state velocities are set.")

    _pot_e = _init_state.getPotentialEnergy()
    _kin_e = _init_state.getKineticEnergy()
    _tot_e = _pot_e + _kin_e

    logger.info(f"Context state energies: kinetic={_kin_e}, potential={_pot_e}, total={_tot_e}")

    _box_volume = _init_state.getPeriodicBoxVolume()
    _bvs = _init_state.getPeriodicBoxVectors()
    _bvs_line = format_box_vectors_line(_bvs)

    logger.info(f"Context state box: volume={_box_volume}, vectors={_bvs_line}")

    return _box_volume, _bvs, _pot_e, _kin_e, _tot_e

# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner):
    """Runner for OpenMM simulations."""

    system: openmm.System
    topology: openmm.app.Topology
    integrator: openmm.Integrator
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
        enforce_box: bool = False,
        get_state_keys: frozenset[str] = GET_STATE_DEFAULT_KEYS,
        openmm_reporter_factories: list[LoggingReporterFactory] | None = None,
    ) -> None:

        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.enforce_box = enforce_box
        self.get_state_keys = get_state_keys

        if openmm_reporter_factories is None or len(openmm_reporter_factories) == 0:
            logger.warning("No OpenMM reporter factories configured.")
        self.openmm_reporter_factories = openmm_reporter_factories if openmm_reporter_factories is not None else []

        self._openmm_reporters = None
        self._init_time = None
        self._pre_cycle_time = None

        self.state_machine = RunnerStateMachine()

        self._report_configuration()

    def _report_configuration(self) -> None:
        logger.info("Details of OpenMMRunner initial configuration")

        logger.info(
            f"OpenMM logging reporters: {', '.join(str(v) for v in self.openmm_reporter_factories)}"
        )

        logger.info(f"Enforce PBCs in getState: {self.enforce_box}")
        logger.info(f"Get state keys: {self.get_state_keys}")

        # system
        num_particles = self.system.getNumParticles()
        num_forces = self.system.getNumForces()
        uses_pbcs = self.system.usesPeriodicBoundaryConditions()
        default_bvs = self.system.getDefaultPeriodicBoxVectors()

        default_bv_volume = triclinic_volume_vec3_quantity(default_bvs)
        default_bv_line = format_box_vectors_line(default_bvs)        

        logger.info(f"System: num_particles={num_particles}, num_forces={num_forces}, uses_pbcs={uses_pbcs}")
        logger.info(f"System default box vectors: volume={default_bv_volume}, vectors={default_bv_line}")

        # topology
        num_chains = self.topology.getNumChains()
        num_residues = self.topology.getNumResidues()
        num_atoms = self.topology.getNumAtoms()
        num_bonds = self.topology.getNumBonds()

        top_bvs = self.topology.getPeriodicBoxVectors()


        logger.info(
            f"Topology: num_chains={num_chains}, num_residues={num_residues}, num_atoms={num_atoms}, num_bonds={num_bonds}"
        )
        if top_bvs is not None:
            top_bv_volume = triclinic_volume_vec3_quantity(default_bvs)
            bv_line = format_box_vectors_line(default_bvs)
            logger.info(
                f"Topology box vectors: volume={top_bv_volume} vectors={bv_line}"
            )
        else:
            logger.info("Topology box vectors not set.")

        chain_ids = [
            chain.id
            for chain
            in self.topology.chains()
        ]
        logger.info(f"Topology Chains (IDs): {','.join(chain_ids)}")

        for chain in self.topology.chains():
            num_residues = len(list(chain.residues()))
            num_atoms = len(list(chain.atoms()))
            logger.info(
                f"Chain {chain.index}: id={chain.id}, num_residues={num_residues}, num_atoms={num_atoms}"
            )

        # integrator
        # UGLY: just dump the XML for simplicity
        integrator_xml = openmm.XmlSerializer.serialize(self.integrator).replace("\n", " ")
        logger.info(f"Integrator: {integrator_xml}")

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
            logger.info(f"Platform instantiated.")

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

        logger.info(f"Time to generate the system: {gen_sim_time:.4f} s")

        logger.info("Information on initial simulation state")
        before_volume, before_bvs, before_pot_e, before_kin_e, before_tot_e = _report_simulation(simulation)

        # actually run the simulation

        steps_start = time.time()

        # Run the simulation segment for the number of time steps
        logger.info("Running MD steps")
        simulation.step(segment_length)

        steps_end = time.time()
        steps_time = steps_end - steps_start

        logger.info(f"Time to run {segment_length} sim steps: {steps_time:.4f} s")

        logger.info("Information on final simulation state")
        after_volume, after_bvs, after_pot_e, after_kin_e, after_tot_e = _report_simulation(simulation)

        _before_lengths = (
            np.linalg.norm(before_bvs[0]),
            np.linalg.norm(before_bvs[1]),
            np.linalg.norm(before_bvs[2]),
        )
        _after_lengths = (
            np.linalg.norm(after_bvs[0]),
            np.linalg.norm(after_bvs[1]),
            np.linalg.norm(after_bvs[2]),
        )

        _delta_lengths = [
            after_length - before_length
            for after_length, before_length
            in zip(_after_lengths, _before_lengths, strict=True)
        ]
        _delta_lengths_line = f"({_delta_lengths[0]}, {_delta_lengths[1]}, {_delta_lengths[2]})"

        # report on the change in energies and box volume
        _delta_volume = after_volume - before_volume
        _delta_pot_e = after_pot_e - before_pot_e
        _delta_kin_e = after_kin_e - before_kin_e
        _delta_tot_e = after_tot_e - before_tot_e

        logger.info(
            f"State changes in Unitcell: box_volume={_delta_volume}, lengths={_delta_lengths_line}, "
        )

        logger.info(
            f"State changes in Energy: potential_E={_delta_pot_e}, kinetic_E={_delta_kin_e}, total_E={_delta_tot_e}"
        )

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
        logger.info(f"Getting context state time: {get_state_time:.4f} s")

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logger.info(f"Total internal run_segment time: {run_segment_time:.4f} s")

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
    enforce_box: bool = False
    get_state_keys: frozenset[str] = attrs.field(default=GET_STATE_DEFAULT_KEYS)
    openmm_reporter_factories: list[LoggingReporterFactory] | None = attrs.field(default=DEFAULT_OPENMM_REPORTER_FACTORIES)

    def __call__(self) -> OpenMMRunner:

        return OpenMMRunner(
            system=copy.deepcopy(self.system),
            topology=copy.deepcopy(self.topology),
            integrator=copy.deepcopy(self.integrator),
            enforce_box=self.enforce_box,
            get_state_keys=self.get_state_keys,
            openmm_reporter_factories=self.openmm_reporter_factories,
        )
