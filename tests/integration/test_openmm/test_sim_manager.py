import functools
import psutil
import openmm
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.work_mapper.serial import SerialMapper
from wepy.work_mapper.openmm import (
    OpenMMSerialWorkMapperFactory,
    OpenMMProcPoolWorkMapperFactory,
)
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

    num_walkers = 4

    walker_states = [
        OpenMMState.from_dwim(
            positions=lj_sys.positions,
        )
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
        work_mapper_factory=OpenMMSerialWorkMapperFactory(
            platform="Reference",
        )
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=100,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMSerialWorkMapperFactory(
            platform="CPU",
            global_platform_properties={"Threads" : "1"},
        )
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=100,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMSerialWorkMapperFactory(
            platform="CPU",
            global_platform_properties={"Threads" : "4"},
        )
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=10000000000,
    )
    
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
        OpenMMState.from_dwim(
            positions=lj_sys.positions,
        )
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

    # As an example of a useful configuration. There are 4 walkers in
    # each cycle so that is the max number of processes that should be
    # used.
    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            platform="Reference",
            num_procs=4,
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=100,
    )

    # As an example of a useful configuration. There are 4 walkers in
    # each cycle so that is the max number of processes that should be
    # used, however for each worker we can assign more threads based
    # on how many CPUs you have. Here we use psutil to reliably get
    # the number of cores and divide that by the number of walkers.
    num_cores = len(psutil.Process().cpu_affinity())
    cores_per_worker = (num_cores // num_walkers)
    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            platform="CPU",
            num_procs=len(walker_states),
            global_platform_properties={"Threads" : str(cores_per_worker)},
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=2,
        segment_lengths=100,
    )

    # Just as way of example of how GPU device specific arguments
    # would work, we assign 1 thread as global but then override that
    # for one specific worker. Note that you need to explicitly
    # enumerate the device IDs then.
    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            platform="CPU",
            num_procs=len(walker_states),
            global_platform_properties={"Threads" : "1"},
            device_ids=[0,1,2,3],
            device_platform_properties=[
                {}, {}, {},
                {"Threads" : "3"}
            ]
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=2,
        segment_lengths=100,
    )
    
# def test_ray_mapper():
#     lj_sys = LennardJonesPair()
#     integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)


#     runner = OpenMMRunner(
#         system=lj_sys.system,
#         topology=lj_sys.topology,
#         integrator=integrator,
#     )

#     num_walkers = 4

#     walker_states = [
#         OpenMMState.from_dwim(
#             positions=lj_sys.positions,
#         )
#         for _
#         in range(num_walkers)
#     ]

#     init_walker_weight = 1 / num_walkers
#     init_walkers = [
#         Walker(
#             state=walker_state,
#             weight=init_walker_weight,
#         )
#         for walker_state
#         in walker_states
#     ]

#     sim_manager = Manager(
#         init_walkers=init_walkers,
#         runner=runner,
#         resampler=NoResampler(),
#         work_mapper_factory=OpenMMRayWorkMapperFactory(
#             platform="Reference",
#             num_procs=1,
#         ),
#     )

#     new_walkers, sim_components  = sim_manager.run_simulation(
#         n_cycles=2,
#         segment_lengths=10,
#     )
