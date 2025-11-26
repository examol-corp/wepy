from wepy.runners.openmm import (
    resolve_state_data_type_enum_values,
    GET_STATE_KWARG_DEFAULTS,
    get_state_fields_present,
    OpenMMRunner,
    OpenMMState,
    gen_sim_state,
    gen_walker_state,
    OpenMMWalker,
    OpenMMCPUWorker,
    OpenMMGPUWorker,
    OpenMMCPUWalkerTaskProcess,
    OpenMMGPUWalkerTaskProcess,
)

import pytest

import numpy as np
import openmm
import openmm.app
import openmm.unit


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
        bvs = unitcell.to_vec3()
        context.setPeriodicBoxVectors(*bvs)

    return context


def n_lj_system(
    num_particles: int,
    mass: openmm.unit.Quantity = (39.9481 * openmm.unit.dalton),
    sigma: openmm.unit.Quantity = (0.3350 * openmm.unit.nanometer),
    epsilon: openmm.unit.Quantity = (0.996 * openmm.unit.kilojoules_per_mole),
) -> openmm.System:

    system = openmm.System()

    # single nonbonded force
    nb_force = openmm.NonbondedForce()
    nb_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    # TODO: add support for cutoffs

    for idx in range(num_particles):
        system.addParticle(mass)
        nb_force.addParticle(
            0.0 * openmm.unit.elementary_charge,
            sigma,
            epsilon,
        )

    system.addForce(nb_force)

    return system


def particle_line(num_particles: int) -> openmm.unit.Quantity:
    """Initialize a 3D coordinate array."""

    return (
        np.array([[float(idx), 0.0, 0.0] for idx in range(num_particles)])
        * openmm.unit.angstrom
    )


ARGON = openmm.app.Element.getBySymbol("Ar")


def n_particle_topology(
    num_particles: int,
    element: openmm.app.Element = ARGON,
) -> openmm.app.Topology:
    """Create a single particle topology from scratch.

    There will only be one chain, and each particle is it's own
    residue.

    Box vectors are never set.
    """

    top = openmm.app.Topology()

    chain = top.addChain()
    for idx in range(num_particles):

        residue = top.addResidue(element.symbol, chain)
        top.addAtom(
            element.symbol,
            element,
            residue,
        )

    return top


@pytest.fixture
def omm_context() -> openmm.Context:

    system = n_lj_system(2)

    coords = particle_line(2)

    ctx = dummy_context(system, coords)

    return ctx


def test_resolve_state_data_type_enum_values():

    assert resolve_state_data_type_enum_values() == {
        "positions": 1,
        "velocities": 2,
        "energy": 8,
        "forces": 4,
        "integrator_parameters": 64,
        "parameter_derivatives": 32,
        "parameters": 16,
    }


def test_get_state_fields_present(omm_context):

    state = omm_context.getState(positions=True)
    assert get_state_fields_present(state) == ["positions"]


def test_gen_sim_state():

    state = gen_sim_state(
        positions=particle_line(2),
        system=n_lj_system(2),
        integrator=openmm.VerletIntegrator(0.002),
    )


def test_gen_walker_state():

    gen_walker_state(
        positions=particle_line(2),
        system=n_lj_system(2),
        integrator=openmm.VerletIntegrator(0.002),
    )


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
            platform_kwargs={"Threads": 1},
        )

        assert runner._cycle_platform == "CPU"
        assert runner._cycle_platform_kwargs == {"Threads": 1}

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
            platform_kwargs={"Threads": 1},
        )

        assert runner._cycle_platform == "CPU"
        assert runner._cycle_platform_kwargs == {"Threads": 1}

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
            platform=Ellipsis, platform_kwargs={"Threads": 1}
        ) == (None, None)

        assert runner._resolve_platform(platform="CPU", platform_kwargs=None) == (
            "CPU",
            None,
        )
        assert runner._resolve_platform(
            platform="CPU", platform_kwargs={"Threads": 1}
        ) == ("CPU", {"Threads": 1})

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.pre_cycle(
            platform="CPU",
            platform_kwargs={"Threads": 1},
        )

        assert runner._resolve_platform(platform=None, platform_kwargs=None) == (
            "CPU",
            {"Threads": 1},
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
        walker = OpenMMWalker(state, 0.1)

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.run_segment(
            walker,
            2,
        )

        walker = runner.run_segment(walker, 2, getState_kwargs={"positions": True})
        assert walker.state["positions"] is not None
        assert walker.state["velocities"] is None

        runner = OpenMMRunner(
            system=system,
            topology=topology,
            integrator=integrator,
        )

        runner.run_segment(
            walker,
            2,
            platform="Reference",
        )


class TestOpenMMCPUWorker:

    pass


class TestOpenMMGPUWorker:
    pass


class TestOpenMMCPUWalkerTaskProcess:
    pass


class TestOpenMMGPUWalkerTaskProcess:
    pass
