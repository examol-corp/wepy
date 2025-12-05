"""Tests for a realistic end to end use case.

OpenMM Runner, REVO and WExplore resamplers, paralell work mappers.

Configurable platforms.

"""
import copy
import openmm
import psutil

import mdtraj
# TODO: use the high-level API imports
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState
from wepy.sim_manager import Manager
from wepy.work_mapper.openmm import OpenMMProcPoolWorkMapperFactory
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.util.mdtraj import mdtraj_to_json_topology

from wepy_tools.systems.alanine_dipeptide import AlanineDipeptideExplicit, AlanineDipeptideRamachandranDistance


def test_alanine_dipeptide_revo_procpool():

    ala_sys = AlanineDipeptideExplicit()

    integrator = openmm.LangevinIntegrator(300.0, 0.002, 0.1)

    runner = OpenMMRunner(
        system=ala_sys.system,
        topology=ala_sys.topology,
        integrator=integrator,
    )

    num_walkers = 4

    init_state = OpenMMState.from_dwim(
            positions=ala_sys.positions,
        )

    # TODO: remove the need to deepcopy and have the components make
    # their own copies if necessary
    walker_states = [
        copy.deepcopy(init_state)
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

    # number of walkers if less then total cores, otherwise the total
    # number of cores
    num_cores = len(psutil.Process().cpu_affinity())
    if num_cores < num_walkers:
        num_workers = num_cores
        cores_per_worker = 1
    else:
        num_workers = num_walkers
        cores_per_worker = (num_workers // num_walkers)

    json_top = mdtraj_to_json_topology(
            mdtraj.Topology.from_openmm(ala_sys.topology)
        )

    distance_metric = AlanineDipeptideRamachandranDistance(json_top)

    resampler = REVOResampler(
        merge_dist=4,
        char_dist=0.1,
        distance=distance_metric,
        # DEBUG
        # num_proc=num_workers,
        num_proc=1,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner=runner,
        resampler=resampler,
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            platform="CPU",
            num_procs=num_workers,
            global_platform_properties={"Threads" : str(cores_per_worker)},
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=100,
    )
