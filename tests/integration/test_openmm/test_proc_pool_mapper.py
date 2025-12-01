import functools
import openmm
import numpy as np
from wepy.runners.openmm import OpenMMRunner, OpenMMState, gen_sim_state
from wepy.work_mapper.proc_pool_mapper import ProcPoolMapper

from wepy_tools.systems.lennard_jones import LennardJonesPair


def test_proc_pool_mapper():

    lj_sys = LennardJonesPair()
    integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)


    runner = OpenMMRunner(
        system=lj_sys.system,
        topology=lj_sys.topology,
        integrator=integrator,
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

    run_func = functools.partial(
        runner.run_segment,
        platform="Reference",
    )
    
    poolmapper = ProcPoolMapper()
    poolmapper.init(run_func, num_workers=2)

    results = poolmapper.map(
        walker_states,
        [10 for _ in range(num_walkers)],
    )
    assert len(results) == 4

    # try different platform with some global platform settings
    run_func = functools.partial(
        runner.run_segment,
        platform="CPU",
        platform_kwargs={"Threads" : "1"},
    )
    
    poolmapper = ProcPoolMapper()
    poolmapper.init(run_func, num_workers=2)

    poolmapper.map(
        walker_states,
        [10 for _ in range(num_walkers)],
    )

    # Different platform args per "worker"

    run_func = functools.partial(
        runner.run_segment,
        platform="CPU",
    )

    poolmapper = ProcPoolMapper()
    poolmapper.init(
        run_func,
        num_workers=2,
        worker_args=[
            {"platform_kwargs" : {"Threads" : "2"}},
            {"platform_kwargs" : {"Threads" : "3"}},
        ]
    )

    poolmapper.map(
        walker_states,
        [10 for _ in range(num_walkers)],
    )
    
