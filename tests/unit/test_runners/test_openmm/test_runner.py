from wepy_tools.systems.lennard_jones import LennardJonesPair
from wepy.runners.openmm.state import (
    dummy_context,
    resolve_state_data_type_enum_values,
    GET_STATE_KWARG_DEFAULTS,
    get_state_fields_present,
    OpenMMState,
    gen_sim_state,
)

from wepy.runner.openmm.runner import (
    OpenMMRunner,
)

import pytest

import numpy as np
import openmm
import openmm.app
import openmm.unit

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
        * openmm.unit.nanometer
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



class TestOpenMMState:

    # TODO: this whole class needs overhauled but I want the other
    # tests before messing with it too much.

    def test___init__(self):

        state = gen_sim_state(
            positions=particle_line(2),
            system=n_lj_system(2),
            integrator=openmm.VerletIntegrator(0.002),
        )

        state = OpenMMState(state)
        # assert "positions" in state
        # assert state._data == {}


class TestOpenMMRunner:

    def test___init__(self):

        system = n_lj_system(2)
        topology = n_particle_topology(2)
        integrator = openmm.VerletIntegrator(0.002)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert not runner.enforce_box
        assert runner.getState_kwargs == dict(GET_STATE_KWARG_DEFAULTS)

        assert OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            enforce_box=True,
        ).enforce_box

        assert OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            enforce_box=False,
            get_state_kwargs={"enforce_box": True},
        ).enforce_box

        assert OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            enforce_box=False,
            get_state_kwargs={"positions": True},
        ).getState_kwargs == {"positions": True}

    def test_pre_cycle(self):

        system = n_lj_system(2)
        topology = n_particle_topology(2)
        integrator = openmm.VerletIntegrator(0.002)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner.pre_cycle()

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner.pre_cycle(
            platform="CPU",
        )

        assert runner._cycle_platform == "CPU"
        assert runner._cycle_platform_kwargs is None

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner.pre_cycle(
            platform="CPU",
            platform_kwargs={"Threads": "1"},
        )

        assert runner._cycle_platform == "CPU"
        assert runner._cycle_platform_kwargs == {"Threads": "1"}

    def test_post_cycle(self):

        system = n_lj_system(2)
        topology = n_particle_topology(2)
        integrator = openmm.VerletIntegrator(0.002)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner.post_cycle()

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

        runner.pre_cycle(
            platform="CPU",
            platform_kwargs={"Threads": "1"},
        )

        assert runner._cycle_platform == "CPU"
        assert runner._cycle_platform_kwargs == {"Threads": "1"}

        runner.post_cycle()

        assert runner._cycle_platform is None
        assert runner._cycle_platform_kwargs is None

    def test__resolve_platform(self):

        system = n_lj_system(2)
        topology = n_particle_topology(2)
        integrator = openmm.VerletIntegrator(0.002)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        assert runner._resolve_platform(platform=None, platform_kwargs=None) == (
            None,
            None,
        )
        assert runner._resolve_platform(platform=Ellipsis, platform_kwargs=None) == (
            None,
            None,
        )
        assert runner._resolve_platform(
            platform=Ellipsis, platform_kwargs={"Threads": "1"}
        ) == (None, None)

        assert runner._resolve_platform(platform="CPU", platform_kwargs=None) == (
            "CPU",
            None,
        )
        assert runner._resolve_platform(
            platform="CPU", platform_kwargs={"Threads": "1"}
        ) == ("CPU", {"Threads": "1"})

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.pre_cycle(
            platform="CPU",
            platform_kwargs={"Threads": "1"},
        )

        assert runner._resolve_platform(platform=None, platform_kwargs=None) == (
            "CPU",
            {"Threads": "1"},
        )

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
            platform="CPU",
        )

        assert runner._resolve_platform(platform=None, platform_kwargs=None) == (
            "CPU",
            None,
        )

    def test_run_segment(self):

        system = n_lj_system(2)
        topology = n_particle_topology(2)
        # TODO: only currently works with LangevinIntegrator
        integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)

        state = gen_sim_state(
            positions=particle_line(2),
            system=n_lj_system(2),
            integrator=openmm.VerletIntegrator(0.002),
        )

        state = OpenMMState(state)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.run_segment(
            state,
            2,
        )

        new_state = runner.run_segment(state, 2, getState_kwargs={"positions": True})
        assert new_state["positions"] is not None
        assert new_state["velocities"] is None

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.run_segment(
            new_state,
            2,
            platform="Reference",
        )


# class TestOpenMMCPUWorker:

#     pass


# class TestOpenMMGPUWorker:
#     pass


# class TestOpenMMCPUWalkerTaskProcess:
#     pass


# class TestOpenMMGPUWalkerTaskProcess:
#     pass

