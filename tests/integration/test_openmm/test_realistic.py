"""Tests for a realistic end to end use case.

OpenMM Runner, REVO and WExplore resamplers, paralell work mappers.

Configurable platforms.

"""
import importlib.resources
import copy
import openmm
import psutil

import mdtraj
# TODO: use the high-level API imports
from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunnerFactory, OpenMMState, HeartBeatLoggingReporterFactory, OpenMMStateWrapper
from wepy.sim_manager import Manager
from wepy.work_mapper.openmm import OpenMMProcPoolWorkMapperFactory
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.runners.openmm.logger import HeartBeatLoggingReporter
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.util.mdtraj import json_to_mdtraj_topology

from wepy_tools.systems.lennard_jones import LennardJonesPair, PairDistance
from wepy_tools.systems.alanine_dipeptide import AlanineDipeptideRamachandranDistance, AlanineDipeptideExplicitSystem

def test_lennard_jones_revo_procpool():

    test_sys = LennardJonesPair()

    integrator = openmm.LangevinIntegrator(300.0, 0.1, 0.002)

    runner_factory = OpenMMRunnerFactory(
        system=test_sys.system,
        topology=test_sys.topology,
        integrator=integrator,
        # For this test we do want heart beat at shorter interval
        openmm_reporter_factories=[
            # heart beat every step
            HeartBeatLoggingReporterFactory(step_interval=2)
        ]
    )

    num_walkers = 4

    init_state = OpenMMState.from_dwim(
            positions=test_sys.positions,
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
            mdtraj.Topology.from_openmm(test_sys.topology)
        )

    distance_metric = PairDistance()

    resampler = REVOResampler(
        merge_dist=4,
        char_dist=0.1,
        distance=distance_metric,
        num_proc=num_workers,
        # num_proc=1,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner_factory=runner_factory,
        # DEBUG
        # resampler=resampler,
        resampler=NoResampler(),
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            # DEBUG
            platform="Reference",
            num_procs=1,
            # platform="CPU",
            # num_procs=num_workers,
            # global_platform_properties={"Threads" : "1"},
            # # global_platform_properties={"Threads" : str(cores_per_worker)},
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=1,
        segment_lengths=10,
    )

def test_alanine_dipeptide_revo_procpool():

    TEMPERATURE = 300. * openmm.unit.kelvin

    ala_sys = AlanineDipeptideExplicitSystem()

    integrator = openmm.LangevinIntegrator(TEMPERATURE, 0.1, 0.002)

    # add the pseudo forces like barostat
    barostat = openmm.MonteCarloBarostat(
        1. * openmm.unit.atmosphere,
        TEMPERATURE,
    )
    ala_sys.system.addForce(barostat)

    runner_factory = OpenMMRunnerFactory(
        system=ala_sys.system,
        topology=ala_sys.topology,
        integrator=integrator,
        # For this test we do want heart beat at shorter interval
        openmm_reporter_factories=[
            # heart beat every step
            HeartBeatLoggingReporterFactory(step_interval=2)
        ]
    )

    num_walkers = 4

    # TODO: remove the need to deepcopy and have the components make
    # their own copies if necessary
    walker_states = [
        copy.deepcopy(ala_sys.state)
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

    distance_metric = AlanineDipeptideRamachandranDistance(ala_sys.json_top)

    resampler = REVOResampler(
        merge_dist=4,
        char_dist=0.1,
        distance=distance_metric,
        num_proc=num_workers,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner_factory=runner_factory,
        # resampler=NoResampler(),
        resampler=resampler,
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            # num_procs=2,
            # platform="Reference",
            platform="CPU",
            num_procs=num_workers,
            global_platform_properties={"Threads" : str(cores_per_worker)},
        ),
    )

    new_walkers, sim_components  = sim_manager.run_simulation(
        n_cycles=2,
        segment_lengths=100,
    )
    
