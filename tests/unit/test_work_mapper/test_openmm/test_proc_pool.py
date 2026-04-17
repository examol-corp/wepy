# Third Party Library
import openmm
import pytest

# First Party Library
from wepy.runners.openmm import (
    OpenMMRunner,
    OpenMMState,
)
from wepy.work_mapper.openmm.proc_pool import (
    OpenMMProcPoolWorkMapper,
)
from wepy_tools.systems.lennard_jones import LennardJonesPair


@pytest.fixture(scope="function")
def openmm_runner() -> OpenMMRunner:

    lj_sys = LennardJonesPair()
    integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)

    runner = OpenMMRunner(
        system=lj_sys.system,
        topology=lj_sys.topology,
        integrator=integrator,
    )
    return runner


@pytest.fixture(scope="function")
def openmm_state() -> OpenMMState:

    lj_sys = LennardJonesPair()

    state = OpenMMState.from_dwim(
        positions=lj_sys.positions,
    )

    return state


class Test_OpenMMProcPoolWorkMapper:

    @pytest.mark.timeout(5)
    @pytest.mark.flaky(reruns=20)
    def test_all(self, openmm_runner):

        lj_sys = LennardJonesPair()

        init_states = [
            OpenMMState.from_dwim(
                positions=lj_sys.positions,
            )
            for _ in range(4)
        ]

        mapper = OpenMMProcPoolWorkMapper(
            platform="CPU",
            num_procs=2,
            device_ids=[0, 1],
            global_platform_properties={"Threads": "1"},
        )
        mapper.init()

        openmm_runner.init()
        openmm_runner.pre_cycle()

        new_states = mapper.map(
            openmm_runner.run_segment,
            init_states,
            [10 for _ in range(len(init_states))],
        )
