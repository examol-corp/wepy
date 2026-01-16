import pytest

import openmm
import openmm.app

from wepy.reporter.openmm import OpenMMRunnerDashboardSection
from wepy.runners.openmm import OpenMMRunnerFactory

from wepy_tools.systems.lennard_jones import LennardJonesPair

STEP_SIZE = 2 * openmm.unit.femtoseconds

@pytest.fixture
def runner_components() -> (
    tuple[openmm.System, openmm.app.Topology, openmm.LangevinIntegrator]
):

    lj_sys = LennardJonesPair()

    integrator = openmm.LangevinIntegrator(300.0, 0.1, STEP_SIZE)

    return lj_sys.system, lj_sys.topology, integrator

class Test_OpenMMRunnerDashboardSection:

    def test___init__(self, runner_components):

        system, topology, integrator = runner_components
        
        section = OpenMMRunnerDashboardSection(
            runner_factory=OpenMMRunnerFactory(
                system=system,
                topology=topology,
                integrator=integrator,
            )
        )

        assert section.runner_name == "OpenMMRunner"
        assert section.step_time == STEP_SIZE
        assert section.walker_total_sampling_time == 0.0 * openmm.unit.microsecond
        assert section.total_sampling_time == 0.0 * openmm.unit.microsecond
