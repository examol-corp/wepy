"""OpenMM molecular dynamics runner with accessory classes.

OpenMM is a library with support for running molecular dynamics
simulations with specific support for fast GPU calculations. The
component based architecture of OpenMM makes it a perfect fit with
wepy.

In addition to the principle OpenMMRunner class there are a few
classes here that make using OpenMM runner more efficient.

First is a WalkerState class (OpenMMState) that wraps the openmm state
object directly, itself is a wrapper around the C++
datastructures. This gives better performance by not performing copies
to a WalkerState dictionary.

Second, is the OpenMMWalker which is identical to the Walker class
except that it enforces the state is an actual instantiation of
OpenMMState. Use of this is optional.

Finally, is the OpenMMGPUWorker class. This is to be used as the
worker type for the WorkerMapper work mapper. This is necessary to
allow passing of the device index to OpenMM for which GPU device to
use.

"""

# Standard Library
from typing import Any, Annotated, TypedDict, NotRequired, Literal, Final, Self, get_args, TypeAlias
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
from wepy.runners.runner import Runner
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.walker import WalkerState
# from wepy.work_mapper.task_mapper import WalkerTaskProcess
# from wepy.work_mapper.worker import Worker

# AtomNDArray = NDArray[Shape["N atoms, 3 dimensions"], Floating]
# BoxVectorsNDArray = NDArray[Shape["3, 3"], Floating]

## Constants

StateFieldName = Literal[
    "positions",
    "velocities",
    "forces",
    "kinetic_energy",
    "potential_energy",
    "time",
    "box_vectors",
    "box_volume",
    "parameters",
    "parameter_derivatives",
]

# OPENMM_STATE_FIELD_KEYS: frozenset[str, ...] = frozenset(get_args(OpenMMStateFields))
# """Names of the fields of the OpenMMState."""
STATE_FIELD_NAMES: frozenset[StateFieldName] = frozenset(get_args(StateFieldName))

FieldDataType: TypeAlias = openmm.UnitQuantity | frozenmap[str, Any]

StateDataTypeName: TypeAlias = Literal[
    "positions",
    "velocities",
    "forces",
    "energy",
    "parameters",
    "parameter_derivatives",
    # NOTE: integrator_parameters show up here but are not accessible
    # as fields on the state
    "integrator_parameters",
]

STATE_DATA_TYPE_ENUM_NAMES = frozenmap(
    {
        "positions": "Positions",
        "velocities": "Velocities",
        "forces": "Forces",
        "energy": "Energy",
        "parameters": "Parameters",
        "parameter_derivatives": "ParameterDerivatives",
        "integrator_parameters": "IntegratorParameters",
    }
)

FIELD_GETTER_NAMES = frozenmap(
    {
        "positions": "getPositions",
        "velocities": "getVelocities",
        "forces": "getForces",
        "kinetic_energy": "getKineticEnergy",
        "potential_energy": "getPotentialEnergy",
        "time": "getTime",
        "box_vectors": "getPeriodicBoxVectors",
        "box_volume": "getPeriodicBoxVolume",
        "parameters": "getParameters",
        "parameter_derivatives": "getEnergyParameterDerivatives",
    }
)

GetStateKeyWords = Literal[
    "positions",
    "velocities",
    "forces",
    "energy",
    "parameters",
    "parameterDerivatives",
]

GET_STATE_KEYWORDS: frozenmap[StateFieldName, GetStateKeyWords | None] = frozenmap({
    "positions": "positions",
    "velocities": "velocities",
    "forces": "forces",
    "kinetic_energy": "energy",
    "potential_energy": "energy",
    "time": None,
    "box_vectors": None,
    "box_volume": None,
    "parameters": "parameters",
    "parameter_derivatives": "parameterDerivatives",
})

GET_STATE_DEFAULT_ENFORCE_PERIODIC_BOX = False


# when we use the get_state function from the simulation context we
# can pass options for what kind of data to get, this is the default
# to get all the data. TODO not really sure what the 'groups' keyword
# is for though
GET_STATE_KWARG_DEFAULTS = frozenmap({
    "getPositions" : True,
    "getVelocities" : True,
    "getForces" : True,
    "getEnergy" : True,
    "getParameters" : True,
    "getParameterDerivatives" : False,
    "enforcePeriodicBox" : True,
})
"""Mapping of key word arguments to the simulation.context.getState
method for retrieving data for a simulation state. By default we set
each as True to retrieve all information. The presence or absence of
them is handled by the OpenMMState.

"""

def get_context_state(
    context: openmm.Context,
    fields: frozenset[StateFieldName] | None = None,
) -> openmm.State:
    """Retrieve a state from a context using field names."""

    if fields is None:
        _fields = STATE_FIELD_NAMES
    else:
        _fields = fields

    kwargs = {}
    for field_name in _fields:

        kwarg = GET_STATE_KEYWORDS[field_name]
        if kwarg is not None:
            kwargs[kwarg] = True

    return context.getState(
        **kwargs,
        enforcePeriodicBox=GET_STATE_DEFAULT_ENFORCE_PERIODIC_BOX,
    )


def resolve_state_data_type_enum_values() -> frozenmap[StateDataTypeName, int]:
    """Gets the enum values for each field in the state.

    These are int values which are used for bitflag operations.
    """
    enum_values = {}
    for our_name, enum_name in STATE_DATA_TYPE_ENUM_NAMES.items():
        enum_values[our_name] = getattr(openmm.State, enum_name)

    return frozenmap(enum_values)


# reversed since that is the order we check them in and is a frequent operation
STATE_DATA_TYPE_ENUM_VALUES: tuple[tuple[StateDataTypeName, int], ...] = tuple(
    sorted(
        [(k, v) for k, v in resolve_state_data_type_enum_values().items()],
        key=lambda x: x[1],
        reverse=True,
    )
)


def get_state_core_fields_present(
    sim_state: openmm.State,
) -> frozenset[StateDataTypeName]:
    """Figure out which core data fields are present in the State.

    This does not include accessory attributes:
      - time
      - box_vectors and box volume
      - energies

    Note that this also includes the 'integrator_parameters' which are
    not accessible from the state getters.
    """

    flag_sum = sim_state.getDataTypes()

    flag_fields = []
    flag_values = []
    flag_cum = flag_sum
    for field_name, flag_value in STATE_DATA_TYPE_ENUM_VALUES:
        if flag_value > flag_cum:
            continue
        elif flag_value == flag_cum:
            flag_fields.append(field_name)
            flag_values.append(flag_value)
            break

        else:
            flag_fields.append(field_name)
            flag_values.append(flag_value)
            flag_cum -= flag_value

    # double check they sum up
    assert sum(flag_values) == flag_sum

    return frozenset(flag_fields)

def get_state_fields_present(sim_state: openmm.State) -> frozenset[StateFieldName]:
    """Figure out which accessible state fields are present in a State.

    This includes all the accessible fields that have getters
    associated with them.

    Notably this excludes the 'integrator_parameters'.
    """

    # get which core fields are present
    present_core_fields = get_state_core_fields_present(sim_state)

    present_fields = set()

    # core fields
    for field in {
        "positions",
        "velocities",
        "forces",
        "parameters",
        "parameter_derivatives",
    }:

        if field in present_core_fields:
            present_fields.add(field)

    # energy is a little different
    if "energy" in present_core_fields:
        present_fields = present_fields | ENERGY_FIELDS

    # handle box fields efficiently
    try:
        sim_state.getPeriodicBoxVectors()
    except openmm.OpenMMException:
        pass
    else:
        present_fields.add("box_vectors")
        present_fields.add("box_volume")

    # then get whether the rest of the accessory fields are available
    try:
        sim_state.getTime()
    except openmm.OpenMMException:
        pass
    else:
        present_fields.add("time")

    return frozenset(present_fields)


# def resolve_state_data_type_enum_values() -> dict[str, int]:
#     enum_values = {}
#     for our_name, enum_name in STATE_DATA_TYPE_ENUM_NAMES.items():
#         enum_values[our_name] = getattr(openmm.State, enum_name)

#     return enum_values


# # reversed since that is the order we check them in and is a frequent operation
# STATE_DATA_TYPE_ENUM_VALUES: list[tuple[str, int]] = list(
#     sorted(
#         [(k, v) for k, v in resolve_state_data_type_enum_values().items()],
#         key=lambda x: x[1],
#         reverse=True,
#     )
# )


# def get_state_fields_present(sim_state: openmm.State) -> list[str]:
#     """For a state returns a set of the field data types present in it."""

#     flag_sum = sim_state.getDataTypes()

#     flag_fields: list[str] = []
#     flag_values: list[int] = []
#     flag_cum: int = flag_sum
#     for field_name, flag_value in STATE_DATA_TYPE_ENUM_VALUES:
#         if flag_value > flag_cum:
#             continue
#         elif flag_value == flag_cum:
#             flag_fields.append(field_name)
#             flag_values.append(flag_value)
#             break

#         else:
#             flag_fields.append(field_name)
#             flag_values.append(flag_value)
#             flag_cum -= flag_value

#     # double check they sum up
#     assert sum(flag_values) == flag_sum

#     return flag_fields


# the Units objects that OpenMM uses internally and are returned from
# simulation data

# TODO: this is never used and we only need the unit names. Its okay
# to use openmm.units here but other runners should use a units sytem
# like pint which is easier to install. So we should remove this since
# its not used.

# UNITS = (('positions_unit', openmm.unit.nanometer),
#          ('time_unit', openmm.unit.picosecond),
#          ('box_vectors_unit', openmm.unit.nanometer),
#          ('velocities_unit', openmm.unit.nanometer/openmm.unit.picosecond),
#          ('forces_unit', openmm.unit.kilojoule / (openmm.unit.nanometer * openmm.unit.mole)),
#          ('box_volume_unit', openmm.unit.nanometer),
#          ('kinetic_energy_unit', openmm.unit.kilojoule / openmm.unit.mole),
#          ('potential_energy_unit', openmm.unit.kilojoule / openmm.unit.mole),
#         )
# """Mapping of units identifiers to the corresponding openmm.units Unit objects."""

# the names of the units from the units objects above. This is used
# for saving them to files
UNIT_NAMES: tuple[tuple[str, str], ...] = (
    ("positions_unit", openmm.unit.nanometer.get_name()),
    ("time_unit", openmm.unit.picosecond.get_name()),
    ("box_vectors_unit", openmm.unit.nanometer.get_name()),
    ("velocities_unit", (openmm.unit.nanometer / openmm.unit.picosecond).get_name()),
    (
        "forces_unit",
        (openmm.unit.kilojoule / (openmm.unit.nanometer * openmm.unit.mole)).get_name(),
    ),
    ("box_volume_unit", openmm.unit.nanometer.get_name()),
    ("kinetic_energy_unit", (openmm.unit.kilojoule / openmm.unit.mole).get_name()),
    ("potential_energy_unit", (openmm.unit.kilojoule / openmm.unit.mole).get_name()),
)
"""Mapping of unit identifier strings to the serialized string spec of the unit."""

# a random seed will be chosen from 1 to RAND_SEED_RANGE_MAX when the
# Langevin integrator is created. 0 is the default and special value
# which will then choose a random value when the integrator is created

# TODO: test this isn't needed
# RAND_SEED_RANGE_MAX = 1000000


class OpenMMStateDict(TypedDict, total=False):
    positions: np.typing.ArrayLike
    velocities: np.typing.ArrayLike
    forces: np.typing.ArrayLike
    kinetic_energy: float
    potential_energy: float
    time: float
    box_vectors: np.typing.ArrayLike
    box_volume: float
    # TODO: parameters

"""OpenMM molecular dynamics runner with accessory classes.

OpenMM is a library with support for running molecular dynamics
simulations with specific support for fast GPU calculations. The
component based architecture of OpenMM makes it a perfect fit with
wepy.

In addition to the principle OpenMMRunner class there are a few
classes here that make using OpenMM runner more efficient.

First is a WalkerState class (OpenMMState) that wraps the openmm state
object directly, itself is a wrapper around the C++
datastructures. This gives better performance by not performing copies
to a WalkerState dictionary.

Second, is the OpenMMWalker which is identical to the Walker class
except that it enforces the state is an actual instantiation of
OpenMMState. Use of this is optional.

Finally, is the OpenMMGPUWorker class. This is to be used as the
worker type for the WorkerMapper work mapper. This is necessary to
allow passing of the device index to OpenMM for which GPU device to
use.

"""

# Standard Library
from typing import Any, Annotated, TypedDict, NotRequired, Literal, Final, Self, get_args, TypeAlias
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
from wepy.runners.runner import Runner
from wepy.util.util import box_vectors_to_lengths_angles
from wepy.walker import WalkerState
# from wepy.work_mapper.task_mapper import WalkerTaskProcess
# from wepy.work_mapper.worker import Worker

# AtomNDArray = NDArray[Shape["N atoms, 3 dimensions"], Floating]
# BoxVectorsNDArray = NDArray[Shape["3, 3"], Floating]

## Constants


# a random seed will be chosen from 1 to RAND_SEED_RANGE_MAX when the
# Langevin integrator is created. 0 is the default and special value
# which will then choose a random value when the integrator is created

# TODO: test this isn't needed
# RAND_SEED_RANGE_MAX = 1000000

PlatformKwargs = dict[str, str]

class OpenMMRunnerSegmentSplitTimes(TypedDict):
    gen_sim_time: float
    steps_time: float
    get_state_time: float
    run_segment_time: float


# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner[OpenMMStateWrapper]):
    """Runner for OpenMM simulations."""

    system: openmm.System
    topology: openmm.app.Topology
    integrator: openmm.Integrator
    platform_name: str
    platform_kwargs: PlatformKwargs
    enforce_box: bool
    getState_kwargs: dict[str, bool]
    # _cycle_platform:
    # _cycle_platform_kwargs:
    _last_cycle_segments_split_times: list[OpenMMRunnerSegmentSplitTimes]

    def __init__(
        self,
        system: openmm.System,
        topology: openmm.app.Topology,
        integrator: openmm.Integrator,
        platform: str | None = None,
        platform_kwargs: PlatformKwargs | None = None,
        enforce_box: bool = False,
        get_state_kwargs: dict[str, bool] | None = None,
    ) -> None:
        """Constructor for OpenMMRunner.

        Parameters
        ----------
        system :
            The system (forcefields) for the simulation.

        topology :
            The topology for you system.

        integrator :
            Integrator for propagating dynamics.

        platform :
            The specification for the default computational platform
            to use. Platform can also be set when run_segment is
            called. If None uses OpenMM default platform, see OpenMM
            documentation for all value but typical ones are:
            Reference, CUDA, OpenCL. If value is None the automatic
            platform determining mechanism in OpenMM will be used.

        platform_kwargs :
            key-values to set for a platform with
            platform.setPropertyDefaultValue as the default for this
            runner.

        enforce_box :
            Calls 'context.getState' with 'enforcePeriodicBox' if True.
             (Default value = False)

        get_state_kwargs :
            key-values to set for getting the state from the OpenMM context.
            keys not included will use the values in GET_STATE_KWARG_DEFAULTS.
            Will override the enforce_box flag.

        Warnings
        --------
        Regarding the enforce_box option.

        When retrieving states from an OpenMM simulation Context, you
        have the option to enforce periodic boundary conditions in the
        resulting atomic positions in a topology aware way that
        doesn't break bonds through boundaries. This is convenient for
        post-processing as this can be a complex task and is not
        readily exposed in the OpenMM API as a standalone function.

        However, in some types of simulations the periodic box vectors
        are ignored (such as implicit solvent ones) despite there
        being no option to not have periodic boundaries in the context
        itself. Likely if you are running one of these kinds of
        simulations you will not pay attention to the box vectors at
        all and the random defaults that exist will be very wrong but
        this incorrectness will not show in a non-wepy simulation with
        openmm unless you are handling the context states
        yourself. Then when you run in wepy the default of True to
        enforce the boxes will be applied and confusingly wrong
        answers will result that are difficult to find root cause of.

        """

        if platform is not None:
            assert isinstance(
                platform, str
            ), f"platform should be a string, not {type(platform)}"

        # we save the different components. However, if we are to make
        # this runner picklable we have to convert the SWIG objects to
        # a picklable form
        self.system = system
        self.integrator = integrator

        # these are not SWIG objects
        self.topology = topology
        self.platform_name = platform
        self.platform_kwargs = platform_kwargs

        self.enforce_box = enforce_box

        self.getState_kwargs = {}
        if get_state_kwargs is not None:
            for k in get_state_kwargs:
                self.getState_kwargs[k] = get_state_kwargs[k]

            # override enforce_box option if specified in get_state_kwargs
            if "enforce_box" in get_state_kwargs:
                self.enforce_box = get_state_kwargs["enforce_box"]

        else:
            self.getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)

        self._cycle_platform = None
        self._cycle_platform_kwargs = None

        # for special monitoring purposes to get split times to debug
        # performance
        self._last_cycle_segments_split_times = []

    def pre_cycle(
        self,
        platform: str | None = None,
        platform_kwargs: PlatformKwargs | None = None,
    ) -> None:
        # choose to use the platform spec in this function call or to
        # use the default one saved in the runner

        # if the platform is given locally use this one
        if platform is not None:
            logger.info(
                f"Setting the platform ({platform}) in the 'pre_cycle' OpenMM Runner call"
                f"with platform kwargs: {platform_kwargs}"
            )
            # set the platform and kwargs for this cycle
            self._cycle_platform = platform
            self._cycle_platform_kwargs = platform_kwargs

        # otherwise we just don't set this and let resolution of
        # platform happen at run segment.
        # each segment split times will get appended to this
        self._last_cycle_segments_split_times = []

    def post_cycle(self) -> None:
        # remove the platform and kwargs for this cycle
        self._cycle_platform = None
        self._cycle_platform_kwargs = None

    def _resolve_platform(
        self,
        platform: str | Literal[Ellipsis] | None,
        platform_kwargs: PlatformKwargs | None,
    ) -> tuple[
        str | None,
        PlatformKwargs | None,
    ]:
        # resolve which platform to use

        # force usage of environmental one
        if platform is Ellipsis:
            platform_name = None
            platform_kwargs = None

        # use the runtime given one
        elif platform is not None:
            platform_name = platform
            platform_kwargs = platform_kwargs

        # if the pre_cycle configured platform is set use this over
        # the default
        elif self._cycle_platform is not None:
            platform_name = self._cycle_platform
            platform_kwargs = self._cycle_platform_kwargs

        # use the default one
        elif self.platform_name is not None:
            platform_name = self.platform_name
            platform_kwargs = self.platform_kwargs

        # if the default is not set fall back to the environmental one
        else:
            platform_name = None
            platform_kwargs = None

        return (
            platform_name,
            platform_kwargs,
        )

    def run_segment(
        self,
        walker_state: OpenMMStateWrapper,
        segment_length: int,
        getState_kwargs: dict[str, bool] | None = None,
        platform: str | Literal[Ellipsis] | None = None,
        platform_kwargs: PlatformKwargs | None = None,
    ) -> OpenMMStateWrapper:
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : The walker for which dynamics will be propagated.

        segment_length : The numerical value that specifies how much dynamics are to be run.

        getState_kwargs : Specify the key-word arguments to pass to
            simulation.context.getState when getting simulation
            states. If None defaults object values.


        platform : The specification for the computational platform to
            use. If None will use the default for the runner and
            ignore platform_kwargs. If Ellipsis forces the use of the
            OpenMM default or environmentally defined platform. See
            OpenMM documentation for all value but typical ones are:
            Reference, CUDA, OpenCL. If value is None the automatic
            platform determining mechanism in OpenMM will be used.

        platform_kwargs : Key-values to set for a platform with
            platform.setPropertyDefaultValue for this segment only.


        Returns
        -------
        new_walker_state : Walker after dynamics was run, only the state should be modified.

        """

        run_segment_start = time.time()

        # set the kwargs that will be passed to getState
        _getState_kwargs = (
            getState_kwargs if getState_kwargs is not None else self.getState_kwargs
        )
        logger.info(f"Default 'getState_kwargs' in runner: {self.getState_kwargs}")
        logger.info(f"'getState_kwargs' passed to 'run_segment' : {getState_kwargs}")

        logger.info(
            "After resolving 'getState_kwargs' that will be used are: "
            f"{_getState_kwargs}"
        )

        gen_sim_start = time.time()

        # make a copy of the integrator for this particular segment
        new_integrator = copy.copy(self.integrator)
        # force setting of random seed to 0, which is a special
        # value that forces the integrator to choose another
        # random number
        new_integrator.setRandomNumberSeed(0)

        ## Platform

        logger.info(f"Default 'platform' in runner: {self.platform_name}")

        logger.info(f"pre_cycle set 'platform' in runner: {self._cycle_platform}")

        logger.info(f"'platform' passed to 'run_segment' : {platform}")

        logger.info(f"Default 'platform_kwargs' in runner: {self.platform_kwargs}")

        logger.info(
            f"pre_cycle set 'platform_kwargs' in runner: {self._cycle_platform_kwargs}"
        )

        logger.info(f"'platform_kwargs' passed to 'run_segment' : {platform_kwargs}")

        platform_name, platform_kwargs = self._resolve_platform(
            platform, platform_kwargs
        )

        logger.info(f"Resolved 'platform' : {platform_name}")

        logger.info(f"Resolved 'platform_kwargs' : {platform_kwargs}")

        # create simulation object

        ## create the platform and customize

        # if a platform was given we use it to make a Simulation object
        if platform_name is not None:
            logger.info("Using platform configured in code.")

            # get the platform by its name to use
            platform = openmm.Platform.getPlatformByName(platform_name)
            logger.info(f"Platform object created: {platform}")

            if platform_kwargs is None:
                platform_kwargs = {}

            # set properties from the kwargs if they apply to the platform
            for key, value in platform_kwargs.items():
                if key in platform.getPropertyNames():
                    logger.info(f"Setting platform property: {key} : {value}")
                    platform.setPropertyDefaultValue(key, value)

                else:
                    warn(
                        f"Platform kwargs given ({key} : {value}) "
                        f"but is not valid for this platform ({platform_name})"
                    )

            # make a new simulation object
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator, platform
            )

        # otherwise just use the default or environmentally defined one
        else:
            logger.info("Using environmental platform.")
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator
            )

        # set the state to the context from the walker
        simulation.context.setState(walker_state.sim_state)

        gen_sim_end = time.time()
        gen_sim_time = gen_sim_end - gen_sim_start

        logger.info("Time to generate the system: {}".format(gen_sim_time))

        # actually run the simulation

        steps_start = time.time()

        # Run the simulation segment for the number of time steps
        simulation.step(segment_length)

        steps_end = time.time()
        steps_time = steps_end - steps_start

        logger.info("Time to run {} sim steps: {}".format(segment_length, steps_time))

        get_state_start = time.time()

        get_state_end = time.time()
        get_state_time = get_state_end - get_state_start
        logger.info("Getting context state time: {}".format(get_state_time))

        # generate the new state
        new_state = OpenMMStateWrapper(simulation.context.getState(**_getState_kwargs))

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logger.info("Total internal run_segment time: {}".format(run_segment_time))

        segment_split_times = OpenMMRunnerSegmentSplitTimes({
            "gen_sim_time": gen_sim_time,
            "steps_time": steps_time,
            "get_state_time": get_state_time,
            "run_segment_time": run_segment_time,
        })

        self._last_cycle_segments_split_times.append(segment_split_times)

        return new_state

    def last_cycle_segments_split_times(self) -> list[OpenMMRunnerSegmentSplitTimes]:

        return copy.deepcopy(self._last_cycle_segments_split_times)




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


PlatformKwargs = dict[str, str]

class OpenMMRunnerSegmentSplitTimes(TypedDict):
    gen_sim_time: float
    steps_time: float
    get_state_time: float
    run_segment_time: float


# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner[OpenMMStateWrapper]):
    """Runner for OpenMM simulations."""

    system: openmm.System
    topology: openmm.app.Topology
    integrator: openmm.Integrator
    platform_name: str
    platform_kwargs: PlatformKwargs
    enforce_box: bool
    getState_kwargs: dict[str, bool]
    # _cycle_platform:
    # _cycle_platform_kwargs:
    _last_cycle_segments_split_times: list[OpenMMRunnerSegmentSplitTimes]

    def __init__(
        self,
        system: openmm.System,
        topology: openmm.app.Topology,
        integrator: openmm.Integrator,
        platform: str | None = None,
        platform_kwargs: PlatformKwargs | None = None,
        enforce_box: bool = False,
        get_state_kwargs: dict[str, bool] | None = None,
    ) -> None:
        """Constructor for OpenMMRunner.

        Parameters
        ----------
        system :
            The system (forcefields) for the simulation.

        topology :
            The topology for you system.

        integrator :
            Integrator for propagating dynamics.

        platform :
            The specification for the default computational platform
            to use. Platform can also be set when run_segment is
            called. If None uses OpenMM default platform, see OpenMM
            documentation for all value but typical ones are:
            Reference, CUDA, OpenCL. If value is None the automatic
            platform determining mechanism in OpenMM will be used.

        platform_kwargs :
            key-values to set for a platform with
            platform.setPropertyDefaultValue as the default for this
            runner.

        enforce_box :
            Calls 'context.getState' with 'enforcePeriodicBox' if True.
             (Default value = False)

        get_state_kwargs :
            key-values to set for getting the state from the OpenMM context.
            keys not included will use the values in GET_STATE_KWARG_DEFAULTS.
            Will override the enforce_box flag.

        Warnings
        --------
        Regarding the enforce_box option.

        When retrieving states from an OpenMM simulation Context, you
        have the option to enforce periodic boundary conditions in the
        resulting atomic positions in a topology aware way that
        doesn't break bonds through boundaries. This is convenient for
        post-processing as this can be a complex task and is not
        readily exposed in the OpenMM API as a standalone function.

        However, in some types of simulations the periodic box vectors
        are ignored (such as implicit solvent ones) despite there
        being no option to not have periodic boundaries in the context
        itself. Likely if you are running one of these kinds of
        simulations you will not pay attention to the box vectors at
        all and the random defaults that exist will be very wrong but
        this incorrectness will not show in a non-wepy simulation with
        openmm unless you are handling the context states
        yourself. Then when you run in wepy the default of True to
        enforce the boxes will be applied and confusingly wrong
        answers will result that are difficult to find root cause of.

        """

        if platform is not None:
            assert isinstance(
                platform, str
            ), f"platform should be a string, not {type(platform)}"

        # we save the different components. However, if we are to make
        # this runner picklable we have to convert the SWIG objects to
        # a picklable form
        self.system = system
        self.integrator = integrator

        # these are not SWIG objects
        self.topology = topology
        self.platform_name = platform
        self.platform_kwargs = platform_kwargs

        self.enforce_box = enforce_box

        self.getState_kwargs = {}
        if get_state_kwargs is not None:
            for k in get_state_kwargs:
                self.getState_kwargs[k] = get_state_kwargs[k]

            # override enforce_box option if specified in get_state_kwargs
            if "enforce_box" in get_state_kwargs:
                self.enforce_box = get_state_kwargs["enforce_box"]

        else:
            self.getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)

        self._cycle_platform = None
        self._cycle_platform_kwargs = None

        # for special monitoring purposes to get split times to debug
        # performance
        self._last_cycle_segments_split_times = []

    def pre_cycle(
        self,
        platform: str | None = None,
        platform_kwargs: PlatformKwargs | None = None,
    ) -> None:
        # choose to use the platform spec in this function call or to
        # use the default one saved in the runner

        # if the platform is given locally use this one
        if platform is not None:
            logger.info(
                f"Setting the platform ({platform}) in the 'pre_cycle' OpenMM Runner call"
                f"with platform kwargs: {platform_kwargs}"
            )
            # set the platform and kwargs for this cycle
            self._cycle_platform = platform
            self._cycle_platform_kwargs = platform_kwargs

        # otherwise we just don't set this and let resolution of
        # platform happen at run segment.
        # each segment split times will get appended to this
        self._last_cycle_segments_split_times = []

    def post_cycle(self) -> None:
        # remove the platform and kwargs for this cycle
        self._cycle_platform = None
        self._cycle_platform_kwargs = None

    def _resolve_platform(
        self,
        platform: str | Literal[Ellipsis] | None,
        platform_kwargs: PlatformKwargs | None,
    ) -> tuple[
        str | None,
        PlatformKwargs | None,
    ]:
        # resolve which platform to use

        # force usage of environmental one
        if platform is Ellipsis:
            platform_name = None
            platform_kwargs = None

        # use the runtime given one
        elif platform is not None:
            platform_name = platform
            platform_kwargs = platform_kwargs

        # if the pre_cycle configured platform is set use this over
        # the default
        elif self._cycle_platform is not None:
            platform_name = self._cycle_platform
            platform_kwargs = self._cycle_platform_kwargs

        # use the default one
        elif self.platform_name is not None:
            platform_name = self.platform_name
            platform_kwargs = self.platform_kwargs

        # if the default is not set fall back to the environmental one
        else:
            platform_name = None
            platform_kwargs = None

        return (
            platform_name,
            platform_kwargs,
        )

    def run_segment(
        self,
        walker_state: OpenMMStateWrapper,
        segment_length: int,
        getState_kwargs: dict[str, bool] | None = None,
        platform: str | Literal[Ellipsis] | None = None,
        platform_kwargs: PlatformKwargs | None = None,
    ) -> OpenMMStateWrapper:
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : The walker for which dynamics will be propagated.

        segment_length : The numerical value that specifies how much dynamics are to be run.

        getState_kwargs : Specify the key-word arguments to pass to
            simulation.context.getState when getting simulation
            states. If None defaults object values.


        platform : The specification for the computational platform to
            use. If None will use the default for the runner and
            ignore platform_kwargs. If Ellipsis forces the use of the
            OpenMM default or environmentally defined platform. See
            OpenMM documentation for all value but typical ones are:
            Reference, CUDA, OpenCL. If value is None the automatic
            platform determining mechanism in OpenMM will be used.

        platform_kwargs : Key-values to set for a platform with
            platform.setPropertyDefaultValue for this segment only.


        Returns
        -------
        new_walker_state : Walker after dynamics was run, only the state should be modified.

        """

        run_segment_start = time.time()

        # set the kwargs that will be passed to getState
        _getState_kwargs = (
            getState_kwargs if getState_kwargs is not None else self.getState_kwargs
        )
        logger.info(f"Default 'getState_kwargs' in runner: {self.getState_kwargs}")
        logger.info(f"'getState_kwargs' passed to 'run_segment' : {getState_kwargs}")

        logger.info(
            "After resolving 'getState_kwargs' that will be used are: "
            f"{_getState_kwargs}"
        )

        gen_sim_start = time.time()

        # make a copy of the integrator for this particular segment
        new_integrator = copy.copy(self.integrator)
        # force setting of random seed to 0, which is a special
        # value that forces the integrator to choose another
        # random number
        new_integrator.setRandomNumberSeed(0)

        ## Platform

        logger.info(f"Default 'platform' in runner: {self.platform_name}")

        logger.info(f"pre_cycle set 'platform' in runner: {self._cycle_platform}")

        logger.info(f"'platform' passed to 'run_segment' : {platform}")

        logger.info(f"Default 'platform_kwargs' in runner: {self.platform_kwargs}")

        logger.info(
            f"pre_cycle set 'platform_kwargs' in runner: {self._cycle_platform_kwargs}"
        )

        logger.info(f"'platform_kwargs' passed to 'run_segment' : {platform_kwargs}")

        platform_name, platform_kwargs = self._resolve_platform(
            platform, platform_kwargs
        )

        logger.info(f"Resolved 'platform' : {platform_name}")

        logger.info(f"Resolved 'platform_kwargs' : {platform_kwargs}")

        # create simulation object

        ## create the platform and customize

        # if a platform was given we use it to make a Simulation object
        if platform_name is not None:
            logger.info("Using platform configured in code.")

            # get the platform by its name to use
            platform = openmm.Platform.getPlatformByName(platform_name)
            logger.info(f"Platform object created: {platform}")

            if platform_kwargs is None:
                platform_kwargs = {}

            # set properties from the kwargs if they apply to the platform
            for key, value in platform_kwargs.items():
                if key in platform.getPropertyNames():
                    logger.info(f"Setting platform property: {key} : {value}")
                    platform.setPropertyDefaultValue(key, value)

                else:
                    warn(
                        f"Platform kwargs given ({key} : {value}) "
                        f"but is not valid for this platform ({platform_name})"
                    )

            # make a new simulation object
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator, platform
            )

        # otherwise just use the default or environmentally defined one
        else:
            logger.info("Using environmental platform.")
            simulation = openmm.app.Simulation(
                self.topology, self.system, new_integrator
            )

        # set the state to the context from the walker
        simulation.context.setState(walker_state.sim_state)

        gen_sim_end = time.time()
        gen_sim_time = gen_sim_end - gen_sim_start

        logger.info("Time to generate the system: {}".format(gen_sim_time))

        # actually run the simulation

        steps_start = time.time()

        # Run the simulation segment for the number of time steps
        simulation.step(segment_length)

        steps_end = time.time()
        steps_time = steps_end - steps_start

        logger.info("Time to run {} sim steps: {}".format(segment_length, steps_time))

        get_state_start = time.time()

        get_state_end = time.time()
        get_state_time = get_state_end - get_state_start
        logger.info("Getting context state time: {}".format(get_state_time))

        # generate the new state
        new_state = OpenMMStateWrapper(simulation.context.getState(**_getState_kwargs))

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logger.info("Total internal run_segment time: {}".format(run_segment_time))

        segment_split_times = OpenMMRunnerSegmentSplitTimes({
            "gen_sim_time": gen_sim_time,
            "steps_time": steps_time,
            "get_state_time": get_state_time,
            "run_segment_time": run_segment_time,
        })

        self._last_cycle_segments_split_times.append(segment_split_times)

        return new_state

    def last_cycle_segments_split_times(self) -> list[OpenMMRunnerSegmentSplitTimes]:

        return copy.deepcopy(self._last_cycle_segments_split_times)


def gen_sim_state(
    positions: np.typing.ArrayLike,
    system: openmm.System,
    integrator: openmm.Integrator,
    getState_kwargs: dict[str, bool] | None = None,
) -> openmm.State:
    """Convenience function for generating an openmm.State object.

    Parameters
    ----------
    positions : arraylike of float
        The positions for the system you want to set

    system : openmm.app.System object

    integrator : openmm.Integrator object

    Returns
    -------
    sim_state : openmm.State object

    """

    # handle the getState_kwargs
    tmp_getState_kwargs = getState_kwargs

    # start with the defaults
    getState_kwargs = dict(GET_STATE_KWARG_DEFAULTS)

    # if there were customizations use them
    if tmp_getState_kwargs is not None:
        getState_kwargs.update(tmp_getState_kwargs)

    # generate a throwaway context, using the reference platform so we
    # don't screw up other platform stuff later in the same process
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, copy.copy(integrator), platform)

    # set the positions
    context.setPositions(positions)

    # then just retrieve it as a state using the default kwargs
    sim_state = context.getState(**getState_kwargs)

    return sim_state


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

