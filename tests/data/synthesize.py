from pathlib import Path

import cyclopts
import logging
import copy

# Third Party Library
import openmm
import psutil

# First Party Library
import mdtraj
import wepy
from wepy_tools.systems.alanine_dipeptide import (
    AlanineDipeptideExplicitSystem,
    AlanineDipeptideRamachandranDistance,
)

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger("tests")


DEFAULT_SAVE_FIELDS = (
    "positions",
    "box_vectors",
    "box_volume",
    "potential_energy",
    "kinetic_energy",
)


app = cyclopts.App()

@app.command
def realistic_hdf5_dialanine_explicit(out_path: Path):

    STEP_SIZE = 2.0 * openmm.unit.femtosecond
    TEMPERATURE = 300.0 * openmm.unit.kelvin
    
    ala_sys = AlanineDipeptideExplicitSystem()

    integrator = openmm.LangevinIntegrator(TEMPERATURE, 0.1, STEP_SIZE)

    # add the pseudo forces like barostat
    barostat = openmm.MonteCarloBarostat(
        1.0 * openmm.unit.atmosphere,
        TEMPERATURE,
    )
    ala_sys.system.addForce(barostat)

    runner_factory = wepy.OpenMMRunnerFactory(
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
        wepy.Walker(
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

    resampler_factory = wepy.REVOResamplerFactory(
        merge_dist=4,
        char_dist=0.1,
        distance_metric=distance_metric,
    )

    mdj_top = mdtraj.Topology.from_openmm(ala_sys.topology)
    protein_idxs = mdj_top.select("protein")
    water_idxs = mdj_top.select("water")

    hdf5_reporter = wepy.WepyHDF5Reporter.from_components(
        file_path=out_path,
        topology=ala_sys.json_top,
        resampler_class=wepy.REVOResampler,
        save_fields=DEFAULT_SAVE_FIELDS + ("velocities",),
        # only require these fields for the initial walkers
        init_walker_save_fields=("positions", "box_vectors",),
        sparse_fields={
            "velocities" : 2,
        },
        main_rep_idxs=protein_idxs,
        all_atoms_rep_freq=2,
        alt_reps={
            "water" : (water_idxs, 2),
        }
    )

    sim_manager = wepy.Manager(
        init_walkers=init_walkers,
        runner_factory=runner_factory,
        # resampler=NoResampler(),
        resampler_factory=resampler_factory,
        work_mapper_factory=wepy.OpenMMProcPoolWorkMapperFactory(
            platform="CPU",
            num_procs=num_workers,
            global_platform_properties={"Threads": str(cores_per_worker)},
        ),
        reporters=[hdf5_reporter],
    )

    cycle_time = 10.0 * openmm.unit.picosecond
    cycle_steps = round(cycle_time / STEP_SIZE)

    new_walkers, sim_components = sim_manager.run_simulation(
        n_cycles=3,
        segment_lengths=cycle_steps,
    )

if __name__ == "__main__":
    app()
