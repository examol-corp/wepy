# Third Party Library
import numpy as np
import openmm
import openmm.unit
import pytest
from immutables import Map as frozenmap
from lxml import etree

# First Party Library
from wepy.runners.openmm.state import (
    OpenMMState,
    OpenMMStateValidationError,
    OpenMMStateWrapper,
    _gen_vec3_element,
    dummy_context,
    get_context_state,
    get_state_core_fields_present,
    get_state_fields_present,
    resolve_state_data_type_enum_values,
    state_to_xml,
)
from wepy_tools.systems.lennard_jones import LennardJonesPair

UNIT_CUBE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_dummy_context():
    lj_sys = LennardJonesPair()

    dummy_context(
        lj_sys.system,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        * openmm.unit.nanometer,
    )

    dummy_context(
        lj_sys.system,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        * openmm.unit.angstrom,
        unitcell=UNIT_CUBE * openmm.unit.nanometer,
    )


def test_resolve_state_data_type_enum_values():

    assert resolve_state_data_type_enum_values() == frozenmap(
        {
            "positions": 1,
            "velocities": 2,
            "forces": 4,
            "energy": 8,
            "parameters": 16,
            "parameter_derivatives": 32,
            "integrator_parameters": 64,
        }
    )


@pytest.fixture
def omm_context() -> openmm.Context:

    lj_sys = LennardJonesPair()

    ctx = dummy_context(lj_sys.system, lj_sys.positions)

    return ctx


def test_get_context_state(omm_context):

    get_context_state(omm_context, {})
    get_context_state(
        omm_context,
        {
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
        },
    )


def test_get_state_core_fields_present(omm_context):

    state = omm_context.getState(positions=True)
    assert get_state_core_fields_present(state) == frozenset({"positions"})

    state = omm_context.getState(positions=True, velocities=True, forces=True)
    assert get_state_core_fields_present(state) == frozenset(
        {"positions", "velocities", "forces"}
    )

    state = omm_context.getState(parameters=True, parameterDerivatives=True)
    assert get_state_core_fields_present(state) == frozenset(
        {
            "parameters",
            "parameter_derivatives",
        }
    )

    state = omm_context.getState(energy=True)
    assert get_state_core_fields_present(state) == frozenset(
        {
            "energy",
        }
    )


def test_get_state_fields_present(omm_context):

    state = omm_context.getState()
    assert get_state_fields_present(state) == frozenset(
        {
            "time",
            "box_vectors",
            "box_volume",
        }
    )

    state = omm_context.getState(positions=True)
    assert get_state_fields_present(state) == frozenset(
        {
            "time",
            "box_vectors",
            "box_volume",
            "positions",
        }
    )

    state = omm_context.getState(positions=True, velocities=True, forces=True)
    assert get_state_fields_present(state) == frozenset(
        {
            "time",
            "box_vectors",
            "box_volume",
            "positions",
            "velocities",
            "forces",
        }
    )

    state = omm_context.getState(parameters=True, parameterDerivatives=True)
    assert get_state_fields_present(state) == frozenset(
        {
            "time",
            "box_vectors",
            "box_volume",
            "parameters",
            "parameter_derivatives",
        }
    )

    state = omm_context.getState(energy=True)
    assert get_state_fields_present(state) == frozenset(
        {
            "time",
            "box_vectors",
            "box_volume",
            "kinetic_energy",
            "potential_energy",
        }
    )


@pytest.mark.usefixtures("omm_context")
class Test_OpenMMStateWrapper:

    def test___init__(self, omm_context):

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
        )

        state_wrapper = OpenMMStateWrapper(state)

        assert state_wrapper.core_fields == frozenset(
            {
                "positions",
                "velocities",
                "forces",
            }
        )

        assert state_wrapper.fields == frozenset(
            {
                "time",
                "box_vectors",
                "box_volume",
                "positions",
                "velocities",
                "forces",
            }
        )

        # test after running some MD
        omm_context.getIntegrator().step(1)

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
            energy=True,
            parameters=True,
            parameterDerivatives=True,
        )

        state_wrapper = OpenMMStateWrapper(state)

        assert state_wrapper.fields == frozenset(
            {
                "time",
                "box_vectors",
                "box_volume",
                "positions",
                "velocities",
                "forces",
                "kinetic_energy",
                "potential_energy",
                "parameters",
                "parameter_derivatives",
            }
        )

    def test_fields_in(self, omm_context):

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
        )

        state_wrapper = OpenMMStateWrapper(state)

        assert state_wrapper.fields_in(
            {
                "time",
                "box_vectors",
                "box_volume",
                "positions",
                "velocities",
                "forces",
            }
        )

        assert not state_wrapper.fields_in({"something"})
        assert not state_wrapper.fields_in({"parameters"})

    def test___contains__(self, omm_context):

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
        )

        state_wrapper = OpenMMStateWrapper(state)

        assert "time" in state_wrapper
        assert "box_vectors" in state_wrapper
        assert "positions" in state_wrapper

    def test___len__(self, omm_context):

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
        )

        state_wrapper = OpenMMStateWrapper(state)
        assert len(state_wrapper) == 6

    def test___getitem__(self, omm_context):

        omm_context.getIntegrator().step(1)

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
            energy=True,
            parameters=True,
            parameterDerivatives=True,
            integratorParameters=True,
        )

        state_wrapper = OpenMMStateWrapper(state)

        # always there

        assert isinstance(state_wrapper["time"], openmm.unit.Quantity)

        # core data
        assert isinstance(state_wrapper["positions"], openmm.unit.Quantity)
        assert isinstance(state_wrapper["velocities"], openmm.unit.Quantity)
        assert isinstance(state_wrapper["forces"], openmm.unit.Quantity)
        assert isinstance(state_wrapper["parameters"], frozenmap)
        assert isinstance(state_wrapper["parameter_derivatives"], frozenmap)

        # extras
        assert isinstance(state_wrapper["potential_energy"], openmm.unit.Quantity)
        assert isinstance(state_wrapper["kinetic_energy"], openmm.unit.Quantity)

        # only when there is a box
        assert isinstance(state_wrapper["box_vectors"], openmm.unit.Quantity)
        assert isinstance(state_wrapper["box_volume"], openmm.unit.Quantity)

    def test___eq__(self, omm_context):
        state1 = omm_context.getState()
        state_wrapper1 = OpenMMStateWrapper(state1)
        state2 = omm_context.getState()
        state_wrapper2 = OpenMMStateWrapper(state2)

        assert state_wrapper1 == state_wrapper2

    def test_to_dict(self, omm_context):

        state = omm_context.getState()

        state_wrapper = OpenMMStateWrapper(state)
        assert set(state_wrapper.to_dict().keys()) == {
            "time",
            "box_vectors",
            "box_volume",
        }

        state = omm_context.getState(
            positions=True,
            velocities=True,
            forces=True,
            energy=True,
            parameters=True,
            parameterDerivatives=True,
            integratorParameters=True,
        )

        state_wrapper = OpenMMStateWrapper(state)
        assert set(state_wrapper.to_dict().keys()) == {
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
        }

    def test_serialize_xml(self, omm_context):

        state = omm_context.getState()

        state_wrapper = OpenMMStateWrapper(state)

        assert openmm.XmlSerializer.serialize(state) == state_wrapper.serialize_xml()

    def test_from_dict(self):

        time = 0.0 * openmm.unit.picosecond
        bvs = UNIT_CUBE * openmm.unit.nanometer

        positions = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        velocities = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        forces = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        d = dict(
            time=time,
            box_vectors=bvs,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )
        assert OpenMMState.from_dict(d) == OpenMMState(**d)

    def test_from_xml(self, omm_context):

        state = omm_context.getState()

        state_xml = openmm.XmlSerializer.serialize(state)

        state_wrapper = OpenMMStateWrapper.from_xml(state_xml)

        assert state_wrapper == OpenMMStateWrapper(state)

    def test_from_dict(self, omm_context):

        lj_sys = LennardJonesPair()
        system = lj_sys.system

        state_d = {
            "time": 0.0 * openmm.unit.seconds,
            "box_vectors": UNIT_CUBE * openmm.unit.nanometer,
        }

        sw = OpenMMStateWrapper.from_dict(system, state_d)
        assert sw is not None
        assert "time" in sw
        assert "box_vectors" in sw
        assert "box_volume" in sw

        state_d = {
            "time": 0.0 * openmm.unit.seconds,
            "box_vectors": UNIT_CUBE * openmm.unit.nanometer,
            "positions": lj_sys.positions,
            "velocities": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond,
            # TODO: to tes this I need a system with a parametrizable
            # force
            #
            # "parameters" : {
            #     "a" : 1.0,
            #     "b" : 2.0
            # }
        }

        sw = OpenMMStateWrapper.from_dict(system, state_d)
        assert sw is not None
        assert "time" in sw
        assert "box_vectors" in sw
        assert "box_volume" in sw
        assert "positions" in sw

        with pytest.raises(OpenMMStateValidationError):
            OpenMMStateWrapper.from_dict(system, {})

        # invalid parameter
        with pytest.raises(OpenMMStateValidationError):
            OpenMMStateWrapper.from_dict(
                system,
                {
                    "parameters": {
                        "a": 1.0,
                    }
                },
            )


class Test_OpenMMState:

    def test__validate_array3ds(self):

        pos1 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        vel1 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        force1 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        good_cases = [
            (None, None, None),
            (pos1, None, None),
            (None, vel1, None),
            (None, None, force1),
            (pos1, vel1, None),
            (pos1, None, force1),
            (None, vel1, force1),
            (pos1, vel1, force1),
        ]

        for c in good_cases:
            assert OpenMMState._validate_array3ds(*c) is True

        pos2 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        vel2 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        force2 = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        bad_cases = [
            (pos1, vel2, None),
            (pos1, None, force2),
            (pos1, vel2, force2),
        ]

        for c in bad_cases:
            with pytest.raises(OpenMMStateValidationError):
                OpenMMState._validate_array3ds(*c)

    def test___init__(self):

        bvs = UNIT_CUBE * openmm.unit.nanometer

        OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=bvs,
        )
        OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=bvs,
            positions=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer,
            velocities=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond,
            forces=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole),
        )

    def test___len__(self):
        assert (
            len(
                OpenMMState(
                    time=0.0 * openmm.unit.picosecond,
                    box_vectors=UNIT_CUBE * openmm.unit.nanometer,
                )
            )
            == 2
        )

        assert (
            len(
                OpenMMState(
                    time=0.0 * openmm.unit.picosecond,
                    box_vectors=UNIT_CUBE * openmm.unit.nanometer,
                    positions=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                        ]
                    )
                    * openmm.unit.nanometer,
                    velocities=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                        ]
                    )
                    * openmm.unit.nanometer
                    / openmm.unit.picosecond,
                    forces=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                        ]
                    )
                    * openmm.unit.kilojoule
                    / (openmm.unit.nanometer * openmm.unit.mole),
                )
            )
            == 5
        )

    def test___contains__(self):
        small_state = OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=UNIT_CUBE * openmm.unit.nanometer,
        )

        assert "time" in small_state
        assert "box_vectors" in small_state
        assert "box_volume" not in small_state
        assert "positions" not in small_state
        assert "velocities" not in small_state
        assert "forces" not in small_state
        assert "kinetic_energy" not in small_state
        assert "potential_energy" not in small_state
        assert "parameters" not in small_state
        assert "parameter_derivatives" not in small_state

        large_state = OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=UNIT_CUBE * openmm.unit.nanometer,
            positions=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer,
            velocities=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond,
            forces=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole),
        )

        assert "time" in large_state
        assert "box_vectors" in large_state
        assert "box_volume" not in large_state
        assert "positions" in large_state
        assert "velocities" in large_state
        assert "forces" in large_state
        assert "kinetic_energy" not in large_state
        assert "potential_energy" not in large_state
        assert "parameters" not in large_state
        assert "parameter_derivatives" not in large_state

    def test___getitem__(self):

        bvs = UNIT_CUBE * openmm.unit.nanometer

        state = OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=bvs,
        )

        assert state["time"] == 0.0 * openmm.unit.picosecond
        assert state["box_vectors"] is not None

        with pytest.raises(KeyError):
            state["invalid"]

        with pytest.raises(ValueError):
            state["positions"]

        state = OpenMMState(
            time=0.0 * openmm.unit.picosecond,
            box_vectors=bvs,
            positions=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer,
            velocities=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond,
            forces=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole),
        )

        assert state["positions"] is not None
        assert state["velocities"] is not None
        assert state["forces"] is not None

    def test_from_state(self, omm_context):
        s = OpenMMState.from_state(omm_context.getState())

        assert s.time is not None
        assert s.box_vectors is not None
        assert s.box_volume is not None

        s = OpenMMState.from_state(omm_context.getState(positions=True))
        assert s.positions is not None

        omm_context.setVelocitiesToTemperature(300.0 * openmm.unit.kelvin)
        s = OpenMMState.from_state(
            omm_context.getState(
                positions=True,
                velocities=True,
            )
        )

        assert s.velocities is not None

        # get some forces to actually use
        omm_context.getIntegrator().step(0)

        s = OpenMMState.from_state(
            omm_context.getState(
                positions=True,
                velocities=True,
                forces=True,
            )
        )
        assert s.forces is not None

        s = OpenMMState.from_state(
            omm_context.getState(
                positions=True,
                velocities=True,
                forces=True,
                energy=True,
                parameters=True,
                parameterDerivatives=True,
            )
        )

        assert s.kinetic_energy is not None
        assert s.potential_energy is not None

        assert s.parameters is not None
        assert s.parameter_derivatives is not None

    def test_from_state_wrapper(self, omm_context):

        sw = OpenMMStateWrapper(
            omm_context.getState(
                positions=True,
                velocities=True,
                forces=True,
                energy=True,
                parameters=True,
                parameterDerivatives=True,
            )
        )
        OpenMMState.from_state_wrapper(sw)

    def test_from_dwim(self):
        positions = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        velocities = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        forces = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        assert OpenMMState.from_dwim(
            box_vectors=None,
            positions=positions,
            velocities=velocities,
            forces=forces,
        ) == OpenMMState(
            time=OpenMMState.DWIM_DEFAULT_TIME,
            box_volume=OpenMMState.DWIM_DEFAULT_BOX_VOLUME,
            box_vectors=OpenMMState.DWIM_DEFAULT_UNITCELL,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )

        bv = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )

        assert OpenMMState.from_dwim(
            box_vectors=bv,
            positions=positions,
        ) == OpenMMState(
            time=OpenMMState.DWIM_DEFAULT_TIME,
            box_volume=OpenMMState.DWIM_DEFAULT_BOX_VOLUME,
            box_vectors=bv,
            positions=positions,
        )

    def test_to_dict(self):

        time = 0.0 * openmm.unit.picosecond
        bvs = UNIT_CUBE * openmm.unit.nanometer

        positions = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        velocities = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        forces = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        os = OpenMMState(
            time=time,
            box_vectors=bvs,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )

        osd = os.to_dict()
        assert set(osd.keys()) == {
            "time",
            "box_vectors",
            "positions",
            "velocities",
            "forces",
        }

    def test_to_state_wrapper(self):

        time = 0.0 * openmm.unit.picosecond
        positions = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
        )
        velocities = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.nanometer
            / openmm.unit.picosecond
        )
        forces = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.kilojoule
            / (openmm.unit.nanometer * openmm.unit.mole)
        )

        unitcell = UNIT_CUBE * openmm.unit.nanometer

        # with optional system
        lj_sys = LennardJonesPair()
        system = lj_sys.system
        os = OpenMMState(
            time=time,
            box_vectors=unitcell,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )

        os.to_state_wrapper(system)

        # without the system
        os = OpenMMState(
            time=time,
            box_vectors=unitcell,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )

        os.to_state_wrapper()


def test__gen_vec3_element():

    root = etree.Element("root")

    el = _gen_vec3_element("position", root, (0, 1, 2))

    assert el.tag == "position"
    assert dict(el.attrib) == {
        "x": "0",
        "y": "1",
        "z": "2",
    }


def test_state_to_xml():

    bvs = UNIT_CUBE * openmm.unit.nanometer

    simple_state = OpenMMState(
        time=0.0 * openmm.unit.picosecond,
        box_vectors=bvs,
    )

    simple_xml = state_to_xml(simple_state)

    assert isinstance(openmm.XmlSerializer.deserialize(simple_xml), openmm.State)

    full_state = OpenMMState(
        time=0.0 * openmm.unit.picosecond,
        box_vectors=bvs,
        positions=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        * openmm.unit.nanometer,
        velocities=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        * openmm.unit.nanometer
        / openmm.unit.picosecond,
        forces=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        * openmm.unit.kilojoule
        / (openmm.unit.nanometer * openmm.unit.mole),
        potential_energy=(1.0 * (openmm.unit.kilojoule / openmm.unit.mole)),
        kinetic_energy=(1.0 * (openmm.unit.kilojoule / openmm.unit.mole)),
    )

    full_xml = state_to_xml(full_state)

    assert isinstance(openmm.XmlSerializer.deserialize(full_xml), openmm.State)
