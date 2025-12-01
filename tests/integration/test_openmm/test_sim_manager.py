import functools
import openmm
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState, gen_sim_state
from wepy.work_mapper.serial import SerialMapper
from wepy.sim_manager import Manager
from wepy.resampling.resamplers.noresampler import NoResampler

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

    init_walker_weight = 1 / num_walkers
    init_walkers = [
        Walker(
            state=walker_state,
            weight=init_walker_weight,
        )
        for walker_state
        in walker_states
    ]

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper=SerialMapper(),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=10,
        num_workers=None,
    )
