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
from typing import Any, Annotated, TypedDict, NotRequired
import logging

logger = logging.getLogger(__name__)
# Standard Library
import time
from warnings import warn
import copy

# Third Party Library
import attrs
import numpy as np

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

KEYS: tuple[str] = (
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
)
"""Names of the fields of the OpenMMState."""

# when we use the get_state function from the simulation context we
# can pass options for what kind of data to get, this is the default
# to get all the data. TODO not really sure what the 'groups' keyword
# is for though
GET_STATE_KWARG_DEFAULTS: tuple[tuple[str, bool]] = (
    ("getPositions", True),
    ("getVelocities", True),
    ("getForces", True),
    ("getEnergy", True),
    ("getParameters", True),
    ("getParameterDerivatives", False),
    ("enforcePeriodicBox", True),
)
"""Mapping of key word arguments to the simulation.context.getState
method for retrieving data for a simulation state. By default we set
each as True to retrieve all information. The presence or absence of
them is handled by the OpenMMState.

"""

STATE_DATA_TYPE_ENUM_NAMES: dict[str, str] = {
    "positions": "Positions",
    "velocities": "Velocities",
    "forces": "Forces",
    "energy": "Energy",
    "parameters": "Parameters",
    "parameter_derivatives": "ParameterDerivatives",
    "integrator_parameters": "IntegratorParameters",
}


def resolve_state_data_type_enum_values() -> dict[str, int]:
    enum_values = {}
    for our_name, enum_name in STATE_DATA_TYPE_ENUM_NAMES.items():
        enum_values[our_name] = getattr(openmm.State, enum_name)

    return enum_values


# reversed since that is the order we check them in and is a frequent operation
STATE_DATA_TYPE_ENUM_VALUES: list[int] = list(
    sorted(
        [(k, v) for k, v in resolve_state_data_type_enum_values().items()],
        key=lambda x: x[1],
        reverse=True,
    )
)


def get_state_fields_present(sim_state: openmm.State) -> list[str]:
    """For a state returns a set of the field data types present in it."""

    flag_sum = sim_state.getDataTypes()

    flag_fields: list[str] = []
    flag_values: list[int] = []
    flag_cum: int = flag_sum
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

    return flag_fields


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
UNIT_NAMES: tuple[tuple[str, str]] = (
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


class OpenMMState(WalkerState):
    """Walker state that wraps an openmm.State object.

    The keys for which values in the state are available are given by
    the KEYS module constant (accessible through the class constant of
    the same name as well).

    Additional fields can be added to these states through passing
    extra kwargs to the constructor. These will be automatically given
    a suffix of "_OTHER" to avoid name clashes.

    """

    KEYS: tuple[str] = KEYS
    """The provided attribute keys for the state."""

    OTHER_KEY_TEMPLATE: str = "{}_OTHER"
    """String formatting template for attributes not set in KEYS."""

    def __init__(
        self,
        sim_state: openmm.State,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor for OpenMMState.

        Parameters
        ----------
        state : openmm.State object
            The simulation state retrieved from the simulation constant.

        kwargs : optional

            Additional attributes to set for the state. Will add the
        "_OTHER" suffix to the keys

        """

        # save the simulation state
        self._sim_state = sim_state

        # probe which data fields it has
        self._sim_state_fields_present = get_state_fields_present(self.sim_state)

        # save additional data if given
        self._data = {}
        for key, value in kwargs.items():
            # if the key is already in the sim_state keys we need to
            # modify it and raise a warning
            if key in self.KEYS:
                warn(
                    "Key {} in kwargs is already taken by this class, renaming to {}".format(
                        self.OTHER_KEY_TEMPLATE
                    ).format(
                        key
                    )
                )

                # make a new key
                new_key = self.OTHER_KEY_TEMPLATE.format(key)

                # set it in the data
                self._data[new_key] = value

            # otherwise just set it
            else:
                self._data[key] = value

    @property
    def sim_state(self) -> openmm.State:
        """The underlying openmm.State object this is wrapping."""
        return self._sim_state

    def __getitem__(self, key: str) -> Any:
        # if this was a key for data not mapped from the OpenMM.State
        # object we use the _data attribute
        if (key not in self.KEYS) and (
            (not key.startswith("parameters"))
            and (not key.startswith("parameter_derivatives"))
        ):
            return self._data[key]

        # otherwise we have to specifically get the correct data and
        # process it into an array from the OpenMM.State
        else:
            if key == "positions":
                return self.positions_values()
            elif key == "velocities":
                return self.velocities_values()
            elif key == "forces":
                return self.forces_values()
            elif key == "kinetic_energy":
                return self.kinetic_energy_value()
            elif key == "potential_energy":
                return self.potential_energy_value()
            elif key == "time":
                return self.time_value()
            elif key == "box_vectors":
                return self.box_vectors_values()
            elif key == "box_volume":
                return self.box_volume_value()

            # handle the parameters differently since they are dictionaries of values
            elif key.startswith("parameters"):
                parameters_dict = self.parameters_values()
                if parameters_dict is None:
                    return None
                else:
                    # TODO: this was an attempt at a general way to do
                    # this but it doesn't work and I only ever need
                    # one nested level, so for now we just implement it that way
                    # return self._get_nested_attr_from_compound_key(key, parameters_dict)

                    param_key = key.split("/")[-1]
                    return parameters_dict[param_key]

            elif key.startswith("parameter_derivatives"):
                pd_dict = self.parameter_derivatives_values()
                if pd_dict is None:
                    return None
                else:
                    return self._get_nested_attr_from_compound_key(key, pd_dict)

    ## Array properties

    # Positions
    @property
    def positions(self) -> Annotated[
        openmm.unit.Quantity | None,
        np.typing.ArrayLike,
    ]:
        """The positions of the state as a numpy array openmm.unit.Quantity object."""

        if "positions" in self._sim_state_fields_present:
            return self.sim_state.getPositions(asNumpy=True)
        else:
            return None

    @property
    def positions_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the positions are in."""
        return self.positions.unit

    def positions_values(self) -> np.typing.ArrayLike | None:
        """The positions of the state as a numpy array in the positions_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """
        return self.positions.value_in_unit(self.positions_unit)

    # Velocities
    @property
    def velocities(self) -> Annotated[openmm.unit.Quantity | None, np.typing.ArrayLike]:
        """The velocities of the state as a numpy array openmm.unit.Quantity object."""

        if "velocities" in self._sim_state_fields_present:
            return self.sim_state.getVelocities(asNumpy=True)
        else:
            return None

    @property
    def velocities_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the velocities are in."""
        return self.velocities.unit

    def velocities_values(self) -> np.typing.ArrayLike | None:
        """The velocities of the state as a numpy array in the velocities_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        velocities = self.velocities
        if velocities is None:
            return None
        else:
            return self.velocities.value_in_unit(self.velocities_unit)

    # Forces
    @property
    def forces(self) -> Annotated[
        openmm.unit.Quantity | None,
        np.typing.ArrayLike,
    ]:
        """The forces of the state as a numpy array openmm.unit.Quantity object."""

        if "forces" in self._sim_state_fields_present:
            return self.sim_state.getForces(asNumpy=True)
        else:
            return None

    @property
    def forces_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the forces are in."""
        return self.forces.unit

    def forces_values(self) -> np.typing.ArrayLike | None:
        """The forces of the state as a numpy array in the forces_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        forces = self.forces
        if forces is None:
            return None
        else:
            return self.forces.value_in_unit(self.forces_unit)

    # Box Vectors
    @property
    def box_vectors(self) -> Annotated[
        openmm.unit.Quantity | None,
        np.typing.ArrayLike,
    ]:
        """The box vectors of the state as a numpy array openmm.unit.Quantity object."""
        try:
            return self.sim_state.getPeriodicBoxVectors(asNumpy=True)
        except:
            warn(
                "Unknown exception handled from `self.sim_state.getPeriodicBoxVectors()`, "
                "this is probably because this attribute is not in the State."
            )
            return None

    @property
    def box_vectors_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the box vectors are in."""
        return self.box_vectors.unit

    def box_vectors_values(self) -> np.typing.ArrayLike | None:
        """The box vectors of the state as a numpy array in the
        box_vectors_unit openmm.unit.Unit. This is what is returned by
        the __getitem__ accessor.

        """

        box_vectors = self.box_vectors
        if box_vectors is None:
            return None
        else:
            return self.box_vectors.value_in_unit(self.box_vectors_unit)

    ## non-array properties

    # Kinetic Energy
    @property
    def kinetic_energy(self) -> Annotated[
        openmm.unit.Quantity | None,
        np.typing.ArrayLike,
    ]:
        """The kinetic energy of the state as a numpy array openmm.unit.Quantity object."""
        try:
            return self.sim_state.getKineticEnergy()
        except:
            warn(
                "Unknown exception handled from `self.sim_state.getKineticEnergy()`, "
                "this is probably because this attribute is not in the State."
            )
            return None

    @property
    def kinetic_energy_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the kinetic energy is in."""
        return self.kinetic_energy.unit

    def kinetic_energy_value(self) -> np.typing.ArrayLike | None:
        """The kinetic energy of the state as a numpy array in the kinetic_energy_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        kinetic_energy = self.kinetic_energy
        if kinetic_energy is None:
            return None
        else:
            return np.array(
                [self.kinetic_energy.value_in_unit(self.kinetic_energy_unit)]
            )

    # Potential Energy
    @property
    def potential_energy(self) -> Annotated[
        openmm.unit.Quantity | None,
        np.typing.ArrayLike,
    ]:
        """The potential energy of the state as a numpy array openmm.unit.Quantity object."""
        try:
            return self.sim_state.getPotentialEnergy()
        except:
            warn(
                "Unknown exception handled from `self.sim_state.getPotentialEnergy()`, "
                "this is probably because this attribute is not in the State."
            )
            return None

    @property
    def potential_energy_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the potential energy is in."""
        return self.potential_energy.unit

    def potential_energy_value(self) -> np.typing.ArrayLike | None:
        """The potential energy of the state as a numpy array in the potential_energy_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        potential_energy = self.potential_energy
        if potential_energy is None:
            return None
        else:
            return np.array(
                [self.potential_energy.value_in_unit(self.potential_energy_unit)]
            )

    # Time
    @property
    def time(self) -> openmm.unit.Quantity | None:
        """The time of the state as a numpy array openmm.unit.Quantity object."""
        try:
            return self.sim_state.getTime()
        except:
            warn(
                "Unknown exception handled from `self.sim_state.getTime()`, "
                "this is probably because this attribute is not in the State."
            )
            return None

    @property
    def time_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the time is in."""
        return self.time.unit

    def time_value(self) -> np.typing.ArrayLike | None:
        """The time of the state as a numpy array in the time_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        time = self.time
        if time is None:
            return None
        else:
            return np.array([self.time.value_in_unit(self.time_unit)])

    # Box Volume
    @property
    def box_volume(self) -> openmm.unit.Quantity | None:
        """The box volume of the state as a numpy array openmm.unit.Quantity object."""
        try:
            return self.sim_state.getPeriodicBoxVolume()
        except:
            warn(
                "Unknown exception handled from `self.sim_state.getPeriodicBoxVolume()`, "
                "this is probably because this attribute is not in the State."
            )
            return None

    @property
    def box_volume_unit(self) -> openmm.unit.Unit:
        """The units (as a openmm.unit.Unit object) the box volume is in."""
        return self.box_volume.unit

    def box_volume_value(self) -> np.typing.ArrayLike | None:
        """The box volume of the state as a numpy array in the box_volume_unit
        openmm.unit.Unit. This is what is returned by the __getitem__
        accessor.

        """

        box_volume = self.box_volume
        if box_volume is None:
            return None
        else:
            return np.array([self.box_volume.value_in_unit(self.box_volume_unit)])

    ## Dictionary properties
    ## Unitless

    # Parameters
    @property
    def parameters(self) -> dict[str, openmm.unit.Quantity] | None:
        """The parameters of the state as a dictionary mapping the names of
        the parameters to their values which are numpy array
        openmm.unit.Quantity objects.

        """

        if "parameters" in self._sim_state_fields_present:
            return self.sim_state.getParameters()
        else:
            return None

    @property
    def parameters_unit(self) -> dict[str, openmm.unit.Unit]:
        """The units for each parameter as a dictionary mapping parameter
        names to their corresponding unit as a openmm.unit.Unit
        object.

        """
        param_units = {key: None for key, val in self.parameters.items()}
        return param_units

    def parameters_values(self) -> dict[str, np.typing.ArrayLike] | None:
        """The parameters of the state as a dictionary mapping the name of the
        parameter to a numpy array in the unit for the parameter of the
        same name in the parameters_unit corresponding
        openmm.unit.Unit object. This is what is returned by the
        __getitem__ accessor using the compound key syntax with the
        prefix 'parameters', e.g. state['parameter/paramA'] for the
        parameter 'paramA'.

        """

        if self.parameters is None:
            return None

        param_arrs = {key: np.array(val) for key, val in self.parameters.items()}

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    # Parameter Derivatives
    @property
    def parameter_derivatives(self) -> dict[str, openmm.unit.Quantity] | None:
        """The parameter derivatives of the state as a dictionary mapping the
        names of the parameters to their values which are numpy array
        openmm.unit.Quantity objects.

        """

        if "parameter_derivatives" in self._sim_state_fields_present:
            return self.sim_state.getEnergyParameterDerivatives()
        else:
            return None

    @property
    def parameter_derivatives_unit(self) -> dict[str, openmm.unit.Unit]:
        """The units for each parameter derivative as a dictionary mapping
        parameter names to their corresponding unit as a
        openmm.unit.Unit object.

        """

        param_units = {key: None for key, val in self.parameter_derivatives.items()}
        return param_units

    def parameter_derivatives_values(self) -> dict[str, np.typing.ArrayLike] | None:
        """The parameter derivatives of the state as a dictionary mapping the
        name of the parameter to a numpy array in the unit for the
        parameter of the same name in the parameters_unit
        corresponding openmm.unit.Unit object. This is what is
        returned by the __getitem__ accessor using the compound key
        syntax with the prefix 'parameter_derivatives',
        e.g. state['parameter_derivatives/paramA'] for the parameter
        'paramA'.

        """

        if self.parameter_derivatives is None:
            return None

        param_arrs = {
            key: np.array(val) for key, val in self.parameter_derivatives.items()
        }

        # return None if there is nothing in this
        if len(param_arrs) == 0:
            return None
        else:
            return param_arrs

    # for the dict attributes we need to transform the keys for making
    # a proper state where all __getitem__ things are arrays
    def _dict_attr_to_compound_key_dict(
        self,
        root_key: str,
        attr_dict: dict[str:Any],
    ) -> dict[str, Any]:
        """Transform a dictionary of values within the compound key 'root_key'
        to a dictionary mapping compound keys to values.

        For example give the root_key 'parameters' and the parameters
        dictionary {'paramA' : 1.234} returns {'parameters/paramA' : 1.234}.

        Parameters
        ----------
        root_key : str
            The compound key prefix
        attr_dict : dict of str : value
            The dictionary with simple keys within the root key namespace.

        Returns
        -------
        compound_key_dict : dict of str : value
            The dictionary with the compound keys.

        """

        key_template = "{}/{}"
        cmpd_key_d = {}
        for key, value in attr_dict.items():
            new_key = key_template.format(root_key, key)
            # if this is a proper feature
            if type(value) == np.ndarray:
                cmpd_key_d[new_key] = value
            elif hasattr(value, "__getitem__"):
                cmpd_key_d.update(self._dict_attr_to_compound_key_dict(new_key, value))
            else:
                raise TypeError("Unsupported attribute type")

        return cmpd_key_d

    def _get_nested_attr_from_compound_key(
        self,
        compound_key: str,
        compound_feat_dict: dict[str, Any],
    ) -> Any:
        """Get arbitrarily deeply nested compound keys from the full
        dictionary tree.

        Parameters
        ----------
        compound_key : str
            Compound key separated by '/' characters

        compound_feat_dict : dict
            Dictionary of arbitrary depth

        Returns
        -------
        value
            Value requested by the key.

        """

        key_components = compound_key.split("/")

        # if there is only one component of the key then it is not
        # really compound, we won't complain just return the
        # "dictionary" if it is not actually a dict like
        if not hasattr(compound_feat_dict, "__getitem__"):
            raise TypeError("Must provide a dict-like with the compound key")

        value = compound_feat_dict[key_components[0]]

        # if the value itself is compound recursively fetch the value
        if hasattr(value, "__getitem__") and len(key_components[1:]) > 0:
            subgroup_key = "/".join(key_components[1:])

            return self._get_nested_attr_from_compound_key(subgroup_key, value)

        elif hasattr(value, "__getitem__") and len(key_components[1:]) < 1:
            raise ValueError("Key does not reference a leaf node of attribute")

        # otherwise we have the right key so return the object
        else:
            return value

    def parameters_features(self) -> dict[str, Any] | None:
        """Returns a dictionary of the parameters with their appropriate
        compound keys. This can be used for placing them in the same namespace
        as the rest of the attributes.
        """

        parameters = self.parameters_values()
        if parameters is None:
            return None
        else:
            return self._dict_attr_to_compound_key_dict("parameters", parameters)

    def parameter_derivatives_features(self) -> dict[str, Any] | None:
        """Returns a dictionary of the parameter derivatives with their appropriate
        compound keys. This can be used for placing them in the same namespace
        as the rest of the attributes.
        """

        parameter_derivatives = self.parameter_derivatives_values()
        if parameter_derivatives is None:
            return None
        else:
            return self._dict_attr_to_compound_key_dict(
                "parameter_derivatives", parameter_derivatives
            )

    def omm_state_dict(self) -> OpenMMStateDict:
        """Return a dictionary with all of the default keys from the wrapped
        openmm.State object
        """

        feature_d = {
            "positions": self.positions_values(),
            "velocities": self.velocities_values(),
            "forces": self.forces_values(),
            "kinetic_energy": self.kinetic_energy_value(),
            "potential_energy": self.potential_energy_value(),
            "time": self.time_value(),
            "box_vectors": self.box_vectors_values(),
            "box_volume": self.box_volume_value(),
        }

        params = self.parameters_features()
        if params is not None:
            feature_d.update(params)

        param_derivs = self.parameter_derivatives_features()
        if param_derivs is not None:
            feature_d.update(param_derivs)

        return feature_d

    def dict(self) -> dict[str, Any]:
        # documented in superclass

        d = {}
        for key, value in self._data.items():
            d[key] = value
        for key, value in self.omm_state_dict().items():
            d[key] = value
        return d

    def to_mdtraj(self, topology: mdtraj.Topology) -> mdtraj.Trajectory:
        """Returns an mdtraj.Trajectory object from this walker's state.

        Parameters
        ----------
        topology : mdtraj.Topology object
            Topology for the state.

        Returns
        -------
        state_traj : mdtraj.Trajectory object

        """

        # resize the time to a 1D vector
        unitcell_lengths, unitcell_angles = box_vectors_to_lengths_angles(
            self.box_vectors
        )
        return mdj.Trajectory(
            np.array([self.positions_values()]),
            unitcell_lengths=[unitcell_lengths],
            unitcell_angles=[unitcell_angles],
            topology=topology,
        )

PlatformKwargs = dict[str, str]

class OpenMMRunnerSegmentSplitTimes(TypedDict):
    gen_sim_time: float
    steps_time: float
    get_state_time: float
    run_segment_time: float


# the runner for the simulation which runs the actual dynamics
class OpenMMRunner(Runner[OpenMMState]):
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
    # _last_cycle_segments_split_times: list[float]

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
        platform: str | type(Ellipsis) | None,
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
        walker_state: OpenMMState,
        segment_length: int,
        getState_kwargs: dict[str, bool] | None = None,
        platform: str | type(Ellipsis) | None = None,
        platform_kwargs: PlatformKwargs = None,
        # UGLY: here to satisfy the interface
        cycle_idx: int = 0,
        walker_idx: int = 0
    ) -> OpenMMState:
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
        new_state = OpenMMState(simulation.context.getState(**_getState_kwargs))

        run_segment_end = time.time()
        run_segment_time = run_segment_end - run_segment_start
        logger.info("Total internal run_segment time: {}".format(run_segment_time))

        segment_split_times = {
            "gen_sim_time": gen_sim_time,
            "steps_time": steps_time,
            "get_state_time": get_state_time,
            "run_segment_time": run_segment_time,
        }

        self._last_cycle_segments_split_times.append(segment_split_times)

        return new_state

    def last_cycle_segments_split_times(self) -> OpenMMRunnerSegmentSplitTimes:

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
