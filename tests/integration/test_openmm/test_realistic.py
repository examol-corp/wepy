"""Tests for a realistic end to end use case.

OpenMM Runner, REVO and WExplore resamplers, paralell work mappers.

Configurable platforms.

"""

# Standard Library
import copy

# Third Party Library
import pytest
import mdtraj
import openmm
import psutil

# First Party Library
from wepy.resampling.resamplers.revo import REVOResamplerFactory
from wepy.runners.openmm import OpenMMRunnerFactory, OpenMMState
from wepy.runners.openmm.runner import (
    _DEFAULT_HEARTBEAT_INTERVAL,
    _DEFAULT_STATE_TIME_INTERVAL,
)
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology

# TODO: use the high-level API imports
from wepy.walker import Walker
from wepy.work_mapper.openmm import OpenMMProcPoolWorkMapperFactory
from wepy_tools.systems.alanine_dipeptide import (
    AlanineDipeptideExplicitSystem,
    AlanineDipeptideRamachandranDistance,
)
from wepy_tools.systems.lennard_jones import LennardJonesPair, PairDistance

STEP_SIZE = 2.0 * openmm.unit.femtosecond
TEMPERATURE = 300.0 * openmm.unit.kelvin

# minimum number of steps to hit the logging reporters, useful just
# for testing the defaults
TIME_INTERVAL_STEPS = round(_DEFAULT_STATE_TIME_INTERVAL / STEP_SIZE)
MIN_INTERVAL_STEPS = (
    TIME_INTERVAL_STEPS
    if TIME_INTERVAL_STEPS > _DEFAULT_HEARTBEAT_INTERVAL
    else _DEFAULT_HEARTBEAT_INTERVAL
)

DEFAULT_CYCLE_TIME = 10. * openmm.unit.picosecond
DEFAULT_CYCLE_STEPS = round(DEFAULT_CYCLE_TIME / STEP_SIZE)

def test_lennard_jones_revo_procpool():

    test_sys = LennardJonesPair()

    integrator = openmm.LangevinIntegrator(TEMPERATURE, 0.1, STEP_SIZE)

    runner_factory = OpenMMRunnerFactory(
        system=test_sys.system,
        topology=test_sys.topology,
        integrator=integrator,
    )

    num_walkers = 48

    init_state = OpenMMState.from_dwim(
        positions=test_sys.positions,
    )

    # TODO: remove the need to deepcopy and have the components make
    # their own copies if necessary
    walker_states = [copy.deepcopy(init_state) for _ in range(num_walkers)]

    init_walker_weight = 1 / num_walkers
    init_walkers = [
        Walker(
            state=walker_state,
            weight=init_walker_weight,
        )
        for walker_state in walker_states
    ]

    # number of walkers if less then total cores, otherwise the total
    # number of cores
    num_cores = len(psutil.Process().cpu_affinity())
    if num_cores < num_walkers:
        num_workers = num_cores
        cores_per_worker = 1
    else:
        num_workers = num_walkers
        cores_per_worker = num_workers // num_walkers

    json_top = mdtraj_to_json_topology(mdtraj.Topology.from_openmm(test_sys.topology))

    distance_metric = PairDistance()

    resampler_factory = REVOResamplerFactory(
        merge_dist=4,
        char_dist=0.1,
        distance_metric=distance_metric,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner_factory=runner_factory,
        resampler_factory=resampler_factory,
        # resampler_factory=NoResampler,
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            # DEBUG
            # platform="Reference",
            # num_procs=1,
            platform="CPU",
            num_procs=num_workers,
            # global_platform_properties={"Threads" : "1"},
            global_platform_properties={"Threads": str(cores_per_worker)},
        ),
    )

    new_walkers, sim_components = sim_manager.run_simulation(
        n_cycles=10,
        segment_lengths=DEFAULT_CYCLE_STEPS,
    )

# disable timeout for this one
@pytest.mark.timeout(timeout=0)
def test_alanine_dipeptide_revo_procpool():

    ala_sys = AlanineDipeptideExplicitSystem()

    integrator = openmm.LangevinIntegrator(TEMPERATURE, 0.1, STEP_SIZE)

    # add the pseudo forces like barostat
    barostat = openmm.MonteCarloBarostat(
        1.0 * openmm.unit.atmosphere,
        TEMPERATURE,
    )
    ala_sys.system.addForce(barostat)

    runner_factory = OpenMMRunnerFactory(
        system=ala_sys.system,
        topology=ala_sys.topology,
        integrator=integrator,
    )

    num_walkers = 10

    # TODO: remove the need to deepcopy and have the components make
    # their own copies if necessary
    walker_states = [copy.deepcopy(ala_sys.state) for _ in range(num_walkers)]

    init_walker_weight = 1 / num_walkers
    init_walkers = [
        Walker(
            state=walker_state,
            weight=init_walker_weight,
        )
        for walker_state in walker_states
    ]

    # number of walkers if less then total cores, otherwise the total
    # number of cores
    num_cores = len(psutil.Process().cpu_affinity())
    if num_cores < num_walkers:
        num_workers = num_cores
        cores_per_worker = 1
    else:
        num_workers = num_walkers
        cores_per_worker = num_workers // num_walkers

    distance_metric = AlanineDipeptideRamachandranDistance(ala_sys.json_top)

    resampler_factory = REVOResamplerFactory(
        merge_dist=4,
        char_dist=0.1,
        distance_metric=distance_metric,
    )

    sim_manager = Manager(
        init_walkers=init_walkers,
        runner_factory=runner_factory,
        # resampler=NoResampler(),
        resampler_factory=resampler_factory,
        work_mapper_factory=OpenMMProcPoolWorkMapperFactory(
            platform="CPU",
            num_procs=num_workers,
            # NOTE,TOREV: in practice not limiting this is just faster
            # and gets better utilization. But for tests we don't want
            # it to eat up all the CPU so we limit it and take
            # longer. In CI we probably want it to use everything
            # though so review this later. This is only true when
            # there are very few walkers though and with more CPU
            # utilization goes way up
            global_platform_properties={"Threads": str(cores_per_worker)},
        ),
    )

    # new_walkers, sim_components = sim_manager.run_simulation(
    #     n_cycles=2,
    #     segment_lengths=MIN_INTERVAL_STEPS * 2 + 10,
    # )

    # short number of steps but many cycles to exercise the pools
    new_walkers, sim_components = sim_manager.run_simulation(
        n_cycles=100,
        segment_lengths=10,
    )
    
