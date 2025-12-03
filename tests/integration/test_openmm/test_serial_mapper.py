from typing import Literal
import functools
import openmm

import attrs

from wepy.runners.openmm import OpenMMRunner, OpenMMState, gen_sim_state, PlatformKwargs
from wepy.work_mapper.serial import SerialMapper
from wepy.work_mapper.base import Task

from wepy_tools.systems.lennard_jones import LennardJonesPair

@attrs.define
class OpenMMTask(Task):

    runner: OpenMMRunner
    segment_length: int
    getState_kwargs: dict[str, bool] | None = None

    def __call__(
            self,
            state: OpenMMState,
            platform: str | type(Ellipsis) | None = None,
            platform_kwargs: PlatformKwargs | None = None,
    ) -> OpenMMState:

        self.runner.run_segment(
            state,
            segment_length=self.segment_length,
            getState_kwargs=self.getState_kwargs,
            platform=platform,
            platform_kwargs=platform_kwargs
        )

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
        OpenMMState(gen_sim_state(
            lj_sys.positions,
            system=lj_sys.system,
            integrator=integrator,
        ))
        for _
        in range(num_walkers)
    ]

    mapper = SerialMapper()
    mapper.init()

    mapper.map(
        [
            OpenMMTask(
                runner=runner,
                segment_length=10,
            )
            for _
            in range(num_walkers)
        ],
        walker_states,
    )

class OpenMMWorkerTask(Task):

    runner: OpenMMRunner
    segment_length: int
    getState_kwargs: dict[str, bool] | None = None
    platform: str | type(Ellipsis) | None = None
    platform_kwargs: PlatformKwargs | None = None

    def __call__(self, state: OpenMMState) -> OpenMMState:

        self.runner.run_segment(
            state,
            segment_length=self.segment_length,
            getState_kwargs=self.getState_kwargs,
            platform=self.platform,
            platform_kwargs=self.platform_kwargs
        )

OpenMMPlatformName = Literal[
    "Reference",
    "CPU",
    "OpenCL",
    "CUDA",
    "HIP",
]
        
class OpenMMPlatformSpec:
    name: OpenMMPlatformName
    properties: dict[str, str]

class OpenMMWorkerSpec:
    platform: OpenMMPlatformSpec
    
class OpenMMSerialMapper(SerialMapper):

    def __init__(
            self,
            worker_specs: dict[int, OpenMMWorkerSpec]| None = None,
    ) -> None:

        self._worker_segment_times: dict[int, list[float]] = {0: []}
        self._worker_specs = worker_specs
        

    def gen_task(self, outer_task: OpenMMTask, task_idx: int) -> OpenMMWorkerTask:

        return OpenMMWorkerTask(
            runner=outer_task.runner,
            segment_length=outer_task.segment_length,
            getState_kwargs=outer_task.getState_kwargs,
            platform
        )

def test_serial_devices():

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

    mapper = OpenMMSerialMapper()
    mapper.init()

    mapper.map(
        [
            OpenMMTask(
                runner=runner,
                segment_length=10,
            )
            for _
            in range(num_walkers)
        ],
        walker_states,
    )
