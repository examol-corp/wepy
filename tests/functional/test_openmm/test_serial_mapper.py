import functools
import openmm
from wepy.runners.openmm import OpenMMRunner, OpenMMState, gen_sim_state
from wepy.work_mapper.serial import SerialMapper

from wepy_tools.systems.lennard_jones import LennardJonesPair


def test_serial_mapper():

    lj_sys = LennardJonesPair()
    integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)


    runner = OpenMMRunner(
        system=lj_sys.system,
        topology=lj_sys.topology,
        integrator=integrator,
    )

    run_func = functools.partial(
        runner.run_segment,
        platform="Reference",
    )

    num_walkers = 4

    walker_states = [
        OpenMMState(gen_sim_state(
            lj_sys.positions,
            system=lj_sys.system,
            integrator=integrator,
        ))
        for _
        in range(num_walkers)
    ]

    mapper = SerialMapper()
    mapper.init(run_func)

    mapper.map(
        walker_states,
        [10 for _ in range(num_walkers)],
    )
