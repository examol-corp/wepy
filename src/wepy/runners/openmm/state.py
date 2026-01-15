# Standard Library
import logging
from collections.abc import Collection
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    TypeAlias,
    TypedDict,
    get_args,
)

# Third Party Library
import attrs
import numpy as np
import numpy.typing
import openmm
import openmm.unit
from immutables import Map as frozenmap
from lxml import etree

# First Party Library
from wepy.core import BugError
from wepy.missing import MISSING
from wepy.util.openmm import array3d_to_vec3
from wepy.walker import WalkerState

logger = logging.getLogger(__name__)


class OpenMMStateValidationError(Exception):
    pass


PREFERRED_UNITS_LUT = frozenmap(
    {
        "length": openmm.unit.nanometer,
        "time": openmm.unit.picosecond,
        "temperature": openmm.unit.kelvin,
        "angle": openmm.unit.degrees,
        "molar_mass": openmm.unit.amu,
        "charge": openmm.unit.elementary_charge,
        "molar_energy": openmm.unit.kilojoule / openmm.unit.mole,
        "energy": openmm.unit.kilojoule,
        "subtance": openmm.unit.mole,
        "velocity": openmm.unit.nanometer / openmm.unit.picosecond,
        "molar_force": (openmm.unit.kilojoule / openmm.unit.nanometer)
        / openmm.unit.mole,
        "molar_energy_density": openmm.unit.kilojoule / openmm.unit.mole,
    }
)

PREFERRED_UNITS = [val for val in PREFERRED_UNITS_LUT.values()]


StateFieldName: TypeAlias = Literal[
    "time",
    "box_vectors",
    "box_volume",
    "positions",
    "velocities",
    "forces",
    "parameters",
    "parameter_derivatives",
    "kinetic_energy",
    "potential_energy",
]

STATE_FIELD_NAMES: frozenset[StateFieldName] = frozenset(get_args(StateFieldName))

CORE_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "positions",
        "velocities",
        "forces",
        "parameters",
        "parameter_derivatives",
    }
)

BOX_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "box_vectors",
        "box_volume",
    }
)

ACCESSORY_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "time",
    }
)

FieldDataType: TypeAlias = openmm.unit.Quantity | frozenmap[str, Any]

STATE_FIELD_TYPES: frozenmap[str, type[FieldDataType]] = frozenmap(
    time=openmm.unit.Quantity,
    box_vectors=openmm.unit.Quantity,
    box_volume=openmm.unit.Quantity,
    positions=openmm.unit.Quantity,
    velocities=openmm.unit.Quantity,
    forces=openmm.unit.Quantity,
    kinetic_energy=openmm.unit.Quantity,
    potential_energy=openmm.unit.Quantity,
    parameters=frozenmap[str, Any],
    parameter_derivatives=frozenmap[str, Any],
)

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

STATE_DATA_TYPE_ENUM_NAMES: frozenmap[str, str] = frozenmap(
    positions="Positions",
    velocities="Velocities",
    forces="Forces",
    energy="Energy",
    parameters="Parameters",
    parameter_derivatives="ParameterDerivatives",
    integrator_parameters="IntegratorParameters",
)

FIELD_GETTER_NAMES: frozenmap[str, str] = frozenmap(
    positions="getPositions",
    velocities="getVelocities",
    forces="getForces",
    kinetic_energy="getKineticEnergy",
    potential_energy="getPotentialEnergy",
    time="getTime",
    box_vectors="getPeriodicBoxVectors",
    box_volume="getPeriodicBoxVolume",
    parameters="getParameters",
    parameter_derivatives="getEnergyParameterDerivatives",
)

UNREQUESTED_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "time",
        "box_volume",
        "box_vectors",
    }
)

ARRAYLIKE_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "positions",
        "velocities",
        "forces",
        "box_vectors",
    }
)

SCALAR_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "kinetic_energy",
        "potential_energy",
        "time",
        "box_volume",
    }
)

MAPPING_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "parameters",
        "parameter_derivatives",
    }
)

ENERGY_FIELDS: frozenset[StateFieldName] = frozenset(
    {
        "kinetic_energy",
        "potential_energy",
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

GET_STATE_KEYWORDS: frozenmap[str, GetStateKeyWords | None] = frozenmap(
    positions="positions",
    velocities="velocities",
    forces="forces",
    kinetic_energy="energy",
    potential_energy="energy",
    time=None,
    box_vectors=None,
    box_volume=None,
    parameters="parameters",
    parameter_derivatives="parameterDerivatives",
)


GET_STATE_DEFAULT_ENFORCE_PERIODIC_BOX = False

OPENMM_DEFAULT_DIMENSION_UNITS: frozenmap[str, openmm.unit.Unit] = frozenmap(
    length=openmm.unit.nanometer,
    time=openmm.unit.picosecond,
    energy=openmm.unit.kilojoule,
    substance=openmm.unit.mole,
)

OPENMM_DEFAULT_UNITS: frozenmap[str, openmm.unit.Unit] = frozenmap(
    positions=openmm.unit.nanometer,
    time=openmm.unit.picosecond,
    box_vectors=openmm.unit.nanometer,
    box_volume=openmm.unit.nanometer**3,
    velocities=openmm.unit.nanometer / openmm.unit.picosecond,
    forces=openmm.unit.kilojoule / openmm.unit.nanometer,
    kinetic_energy=openmm.unit.kilojoule,
    potential_energy=openmm.unit.kilojoule,
)


class StateFieldData(TypedDict):
    time: openmm.unit.Quantity
    box_vectors: openmm.unit.Quantity
    box_volume: openmm.unit.Quantity

    positions: NotRequired[openmm.unit.Quantity]
    velocities: NotRequired[openmm.unit.Quantity]
    forces: NotRequired[openmm.unit.Quantity]
    kinetic_energy: NotRequired[openmm.unit.Quantity]
    potential_energy: NotRequired[openmm.unit.Quantity]
    parameters: NotRequired[frozenmap[str, Any]]
    parameter_derivatives: NotRequired[frozenmap[str, Any]]


class StateFieldDataInput(TypedDict):
    time: openmm.unit.Quantity
    box_vectors: openmm.unit.Quantity

    positions: NotRequired[openmm.unit.Quantity]
    velocities: NotRequired[openmm.unit.Quantity]
    parameters: NotRequired[frozenmap[str, Any]]


STATE_REQUIRED_INPUT_FIELDS: frozenset[StateFieldName] = frozenset(
    {"time", "box_vectors"}
)


def dummy_context(
    system: openmm.System,
    positions: openmm.unit.Quantity,
    unitcell: openmm.unit.Quantity | None = None,
) -> openmm.Context:
    """Create a throwaway OpenMM context.

    This uses some hardcoded integrators, etc. to be able to get a
    context which is useful for generating OpenMM objects without
    running any calculations. You can also use it for a simulation but
    it won't do anything meaningful.
    """

    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    if unitcell is not None:
        context.setPeriodicBoxVectors(*unitcell)

    return context


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


def resolve_state_data_type_enum_values() -> frozenmap[str, int]:
    """Gets the enum values for each field in the state.

    These are int values which are used for bitflag operations.
    """
    enum_values = {}
    for our_name, enum_name in STATE_DATA_TYPE_ENUM_NAMES.items():
        enum_values[our_name] = getattr(openmm.State, enum_name)

    return frozenmap(enum_values)


# reversed since that is the order we check them in and is a frequent operation
STATE_DATA_TYPE_ENUM_VALUES: tuple[tuple[str, int], ...] = tuple(
    sorted(
        [(k, v) for k, v in resolve_state_data_type_enum_values().items()],
        key=lambda x: x[1],
        reverse=True,
    )
)


def get_state_core_fields_present(
    sim_state: openmm.State,
) -> frozenset[str]:
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


## Wrapper for a openmm.State


class OpenMMStateWrapper(WalkerState):
    """Useful wrapper around an openmm.State object.

    This object is meant to be an alternative interface to the
    openmm.State object and thus supports the same use case of efficient
    and fine grained data transfer between the openmm.Context (possible
    GPU memory) and local memory.
    """

    state: openmm.State
    core_fields: frozenset[StateDataTypeName]
    fields: frozenset[StateFieldName]

    def __init__(self, state: openmm.State) -> None:

        self.state = state

        # probe which data fields it has
        self.core_fields = get_state_core_fields_present(self.state)
        self.fields = get_state_fields_present(self.state)

    def fields_in(self, fields: Collection[StateFieldName]) -> bool:
        """Checks if all the fields specified are in the state."""

        return len(set(fields) - self.fields) == 0

    def __str__(self) -> str:

        fields_s = ", ".join(self.fields)
        return f"StateWrapper[{fields_s}]"

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, item: str) -> bool:
        return self.fields_in((item,))

    def __getitem__(self, key) -> FieldDataType:
        # if this was a key for data not mapped from the OpenMM.State
        # object we use the _data attribute
        if key not in STATE_FIELD_NAMES:
            raise KeyError(f"Key '{key}' is not a valid key for a state.")

        elif key in STATE_DATA_TYPE_ENUM_NAMES and key not in self.fields:
            raise KeyError(f"Core data field '{key}' is not present in this state")
        else:

            # resolve the getter function for the field
            getter_name = FIELD_GETTER_NAMES[key]

            getter_method = getattr(self.state, getter_name, None)

            if getter_method is None:
                raise BugError(f"No getter ('{getter_name}') for key '{key}'")

            elif key in ARRAYLIKE_FIELDS:
                field_val = getter_method(asNumpy=True)
            elif key in SCALAR_FIELDS:
                field_val = getter_method()
            elif key in MAPPING_FIELDS:
                field_val = frozenmap(dict(getter_method()))
            else:
                raise BugError(f"Unhandled field key {key}, getter '{getter_name}'")

            return field_val

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, type(self)):
            return False

        if self.fields == other.fields:
            for field_name in self.fields & (SCALAR_FIELDS | MAPPING_FIELDS):
                if self[field_name] != other[field_name]:
                    return False

            for field_name in self.fields & (ARRAYLIKE_FIELDS - {"box_vectors"}):
                if not np.array_equal(self[field_name], other[field_name]):
                    return False

        else:
            return False

        return True

    # Methods for the OpenMMBasicStateProtocol
    def get_positions(self) -> openmm.unit.Quantity | None:
        return self["positions"]

    def get_unitcell(self) -> openmm.unit.Quantity | None:
        return self["unitcell"]

    def to_dict(self) -> StateFieldData:

        return StateFieldData({field: self[field] for field in self.fields})

    def serialize_xml(self) -> str:
        """Serialize the openmm.State to openmm XML format."""

        return openmm.XmlSerializer.serialize(self.state)

    @classmethod
    def from_xml(cls, xml_str: str) -> Self:
        """Serialize the openmm.State to openmm XML format."""

        state = openmm.XmlSerializer.deserialize(xml_str)
        return cls(state)

    @classmethod
    def from_dict(
        cls,
        system: openmm.System,
        state_dict: StateFieldDataInput,
    ) -> Self:
        """Convert an input dictionary to a state wrapper and State object.

        Note that the input data structure is slightly different than
        the state output.

        For example the box_volume cannot be provided as an input
        since it is a derived value from the box vectors.

        See the data structure type for inputs.

        For parameters the parameter must be defined in the system forces.
        """

        if (
            len(missing_fields := STATE_REQUIRED_INPUT_FIELDS - set(state_dict.keys()))
            > 0
        ):

            missing_fields_str = ", ".join(missing_fields)

            raise OpenMMStateValidationError(
                f"Missing required fields: {missing_fields_str}"
            )

        # a dummy context used to generate a state only
        ctx = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds),
            openmm.Platform.getPlatformByName("Reference"),
        )

        # the fields which are always in a state dict
        ctx.setTime(state_dict["time"].in_units_of(OPENMM_DEFAULT_UNITS["time"]))

        bvs_vec3 = (
            tuple(v for v in array3d_to_vec3(state_dict["box_vectors"]))
            * state_dict["box_vectors"].unit
        )
        ctx.setPeriodicBoxVectors(*bvs_vec3)

        if "positions" in state_dict:

            ctx.setPositions(state_dict["positions"])

        if "velocities" in state_dict:

            ctx.setVelocities(state_dict["velocities"])

        if "parameters" in state_dict:
            for name, value in state_dict["parameters"].items():
                try:
                    ctx.setParameter(name, value)
                except openmm.OpenMMException:
                    raise OpenMMStateValidationError(
                        f"Could not set parameter '{name}' as there is no matching parameter in the system forces."
                    )

        state = get_context_state(ctx, frozenset(state_dict.keys()))

        return cls(state)


# A plain data structure state


def _maybe_array_equal(
    arr0: openmm.unit.Quantity | None,
    arr1: openmm.unit.Quantity | None,
) -> bool:
    """Similar to numpy.array_equal for quantities."""

    if arr0 is None and arr1 is None:
        return True
    elif arr0 is None or arr1 is None:
        return False
    else:
        return np.array_equal(arr0, arr1)


def _gen_unit_cube() -> numpy.typing.ArrayLike:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@attrs.define
class OpenMMState(WalkerState):
    """Pure data type for an OpenMM state.

    This type does no wrapping and should be serializable as is.

    This is not mapped to a particular openmm.State object but can be
    converted to it.
    """

    time: openmm.unit.Quantity = attrs.field(
        eq=attrs.cmp_using(np.array_equal, require_same_type=False),
    )
    box_vectors: openmm.unit.Quantity = attrs.field(
        eq=attrs.cmp_using(np.array_equal, require_same_type=False),
    )

    box_volume: openmm.unit.Quantity | None = None

    positions: openmm.unit.Quantity | None = attrs.field(
        default=None,
        eq=attrs.cmp_using(_maybe_array_equal, require_same_type=False),
    )
    velocities: openmm.unit.Quantity | None = attrs.field(
        default=None,
        eq=attrs.cmp_using(_maybe_array_equal, require_same_type=False),
    )
    forces: openmm.unit.Quantity | None = attrs.field(
        default=None,
        eq=attrs.cmp_using(_maybe_array_equal, require_same_type=False),
    )
    kinetic_energy: openmm.unit.Quantity | None = None
    potential_energy: openmm.unit.Quantity | None = None
    parameters: frozenmap[str, Any] | None = None
    parameter_derivatives: frozenmap[str, Any] | None = None

    # DWIM: "do what I mean" which corresponds to a common default
    # that is somewhat arbitrary but typically the ergonomic single
    # way to do something
    DWIM_DEFAULT_TIME: ClassVar[openmm.unit.Quantity] = 0 * openmm.unit.picosecond
    DWIM_DEFAULT_UNITCELL: ClassVar[openmm.unit.Quantity] = (
        _gen_unit_cube() * openmm.unit.nanometer
    )
    DWIM_DEFAULT_BOX_VOLUME: ClassVar[openmm.unit.Quantity] = 0.0 * (
        openmm.unit.nanometer**3
    )

    @staticmethod
    def _validate_array3ds(
        positions: openmm.unit.Quantity | None,
        velocities: openmm.unit.Quantity | None,
        forces: openmm.unit.Quantity | None,
    ) -> bool:

        maybe_nums = {
            "positions": positions.shape[0] if positions is not None else None,
            "velocities": velocities.shape[0] if velocities is not None else None,
            "forces": forces.shape[0] if forces is not None else None,
        }

        # check dependending on which attributes are actually given
        which_given = [name for name, e in maybe_nums.items() if e is not None]
        if len(which_given) < 2:
            return True

        elif (
            len(which_given) == 2
            and maybe_nums[which_given[0]] != maybe_nums[which_given[1]]
        ):
            report = ", ".join(f"{name}={maybe_nums[name]}" for name in which_given)

            raise OpenMMStateValidationError(
                f"The number of particles do not match: {report}"
            )

        elif len(which_given) == 3 and (
            maybe_nums[which_given[0]] != maybe_nums[which_given[1]]
            or maybe_nums[which_given[0]] != maybe_nums[which_given[2]]
        ):

            report = ", ".join(f"{name}={maybe_nums[name]}" for name in which_given)

            raise OpenMMStateValidationError(
                f"The number of particles do not match: {report}"
            )

        else:
            return True

    def __attrs_post_init__(self) -> None:

        # validate that at least one field is given
        if not any(f is not None for f in attrs.astuple(self)):
            raise OpenMMStateValidationError(
                "At least one field must be provided to construct the object."
            )

        # validate that coordinates all have the same number of particles
        self._validate_array3ds(self.positions, self.velocities, self.forces)

        # TODO: These aren't critical so avoiding this for now.
        #
        # validate that the box_volume matches the vectors
        #
        # validate that the parameters and derivatives have the same keys

    def __len__(self) -> int:

        count = 0
        for key in STATE_FIELD_NAMES:
            if getattr(self, key, MISSING) is not None:
                count += 1

        return count

    def __contains__(self, key: str) -> bool:

        if key not in STATE_FIELD_NAMES:
            return False

        elif getattr(self, key, MISSING) is None:
            return False

        else:
            return True

    def __getitem__(self, key: str) -> openmm.unit.Quantity:

        if key not in STATE_FIELD_NAMES:
            raise KeyError(f"Field {key} is not a valid OpenMMState key")

        elif getattr(self, key, MISSING) is None:
            raise ValueError(f"Field {key} has no value.")

        else:
            return getattr(self, key, None)

    @classmethod
    def from_dict(cls, data_dict: StateFieldData) -> Self:
        return cls(**data_dict)

    @classmethod
    def from_dwim(
        cls,
        positions: openmm.unit.Quantity,
        time: openmm.unit.Quantity | None = None,
        box_volume: openmm.unit.Quantity | None = None,
        box_vectors: openmm.unit.Quantity | None = None,
        velocities: openmm.unit.Quantity | None = None,
        forces: openmm.unit.Quantity | None = None,
        kinetic_energy: openmm.unit.Quantity | None = None,
        potential_energy: openmm.unit.Quantity | None = None,
        parameters: frozenmap[str, Any] | None = None,
        parameter_derivatives: frozenmap[str, Any] | None = None,
    ):

        return cls(
            time=(cls.DWIM_DEFAULT_TIME if time is None else time),
            box_volume=(
                cls.DWIM_DEFAULT_BOX_VOLUME if box_volume is None else box_volume
            ),
            positions=positions,
            box_vectors=(
                cls.DWIM_DEFAULT_UNITCELL if box_vectors is None else box_vectors
            ),
            velocities=velocities,
            forces=forces,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            parameters=parameters,
            parameter_derivatives=parameter_derivatives,
        )

    @classmethod
    def from_state_wrapper(cls, state_wrapper: OpenMMStateWrapper) -> Self:
        return cls.from_dict(state_wrapper.to_dict())

    @classmethod
    def from_state(cls, state: openmm.State) -> Self:
        return cls.from_state_wrapper(OpenMMStateWrapper(state))

    def dict(self) -> StateFieldData:
        return StateFieldData(
            {
                k: v
                for k, v in attrs.asdict(self, recurse=False).items()
                if v is not None
            }
        )

    def to_state_wrapper(
        self,
        system: openmm.System | None = None,
    ) -> OpenMMStateWrapper:
        """Convert this state to a real wrapped openmm.State.

        Notes:
          If the optional system is provided this provides a more direct
          route for translation.

          If not the state will be serialized and deserialized in memory
          to get a state.
        """

        if system is not None:
            wrapper = OpenMMStateWrapper.from_dict(system, self.dict())

        else:

            xml_str = state_to_xml(self)
            state = openmm.XmlSerializer.deserialize(xml_str)
            wrapper = OpenMMStateWrapper(state)

        return wrapper


def _gen_vec3_element(
    name: str, parent: etree.Element, vec: tuple[int, int, int]
) -> etree.Element:

    return etree.SubElement(
        parent,
        name,
        x=str(vec[0]),
        y=str(vec[1]),
        z=str(vec[2]),
    )


def state_to_xml(
    state: OpenMMState,
    step_count: int = 0,
) -> str:
    """Convert to an OpenMM State XML without the need of a system/context."""

    _time_str = str(state.time.in_units_of(PREFERRED_UNITS_LUT["time"]))
    _step_count_str = str(step_count)

    state_el = etree.Element(
        "State",
        openmmVersion=openmm.version.short_version,
        stepCount=_step_count_str,
        time=_time_str,
        type="State",
        version="1",
    )

    # box vectors
    box_vectors_el = etree.SubElement(
        state_el,
        "PeriodicBoxVectors",
    )

    _bvecs = [
        tuple(vec.in_units_of(PREFERRED_UNITS_LUT["length"]))
        for vec in tuple(state.box_vectors[i] for i in range(3))
    ]

    for name, vec in zip(("A", "B", "C"), _bvecs, strict=True):

        _gen_vec3_element(
            name,
            box_vectors_el,
            vec,
        )

    if state.positions is not None:

        positions_el = etree.SubElement(
            state_el,
            "Positions",
        )

        _positions = state.positions.in_units_of(PREFERRED_UNITS_LUT["length"])

        for atom_vec in _positions:

            _gen_vec3_element(
                "Position",
                positions_el,
                tuple(atom_vec),
            )

    if state.velocities is not None:

        velocities_el = etree.SubElement(
            state_el,
            "Velocities",
        )

        _velocities = state.velocities.in_units_of(PREFERRED_UNITS_LUT["velocity"])

        for atom_vec in _velocities:

            _gen_vec3_element(
                "Velocity",
                velocities_el,
                tuple(atom_vec),
            )

    if state.forces is not None:

        forces_el = etree.SubElement(
            state_el,
            "Forces",
        )

        _forces = state.forces.in_units_of(PREFERRED_UNITS_LUT["molar_force"])

        for atom_vec in _forces:

            _gen_vec3_element(
                "Force",
                forces_el,
                tuple(atom_vec),
            )

    if state.kinetic_energy is not None or state.potential_energy is not None:

        energies_el = etree.SubElement(
            state_el,
            "Energies",
            **(
                {
                    "KineticEnergy": str(
                        state.kinetic_energy.in_units_of(
                            PREFERRED_UNITS_LUT["molar_energy_density"]
                        )
                    )
                }
                if state.kinetic_energy is not None
                else {}
            ),
            **(
                {
                    "PotentialEnergy": str(
                        state.potential_energy.in_units_of(
                            PREFERRED_UNITS_LUT["molar_energy_density"]
                        )
                    )
                }
                if state.potential_energy is not None
                else {}
            ),
        )

    # TODO: parameters, parameter derivatives. I have no working
    # example on how to do this so I am eliding them.
    if state.parameters is not None or state.parameter_derivatives is not None:

        logger.warning(
            "A state was provided to the XML serializer with parameters or parameter_derivatives, but these are currently not serialized."
        )

    xml_str = etree.tostring(
        state_el,
        xml_declaration=True,
        pretty_print=True,
    )

    return xml_str.decode()
