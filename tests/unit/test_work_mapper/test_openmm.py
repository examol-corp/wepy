from wepy.runners.openmm import (
    OpenMMRunner,
    OpenMMState,
    gen_sim_state,
)
from wepy.work_mapper.openmm import (
    OpenMMTask,
    OpenMMSerialWorkMapper,
    OpenMMProcPoolWorkMapper,
    OpenMMRayPoolWorkMapper,
)

import pytest

import numpy as np
import openmm
import openmm.app
import openmm.unit

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
    
    state = OpenMMState(gen_sim_state(
            lj_sys.positions,
            system=lj_sys.system,
            integrator=openmm.VerletIntegrator(0.002),
    ))

    return state
    


class TestOpenMMTask:

    def test___call__(self, openmm_runner, openmm_state):

        task = OpenMMTask(
            runner=openmm_runner,
            segment_length=2,
        )

        new_state = task(openmm_state)

class TestOpenMMSerialWorkMapper:

    def test_all(self, openmm_runner):

        lj_sys = LennardJonesPair()

        init_states = [
            OpenMMState(gen_sim_state(
                        lj_sys.positions,
                        system=lj_sys.system,
                        integrator=openmm.VerletIntegrator(1),
                ))
            for _
            in range(4)
        ]

        mapper = OpenMMSerialWorkMapper(
            platform="CPU",
            global_platform_properties={"Threads" : "2"},
        )

        mapper.init()
        new_states = mapper.map(
            [
                OpenMMTask(
                    runner=openmm_runner,
                    segment_length=2,
                )
                for _ in range(len(init_states))
            ],
            init_states,
        )
        
class TestOpenMMProcPoolWorkMapper:
    def test_all(self, openmm_runner):

        lj_sys = LennardJonesPair()
        
        init_states = [
            OpenMMState(gen_sim_state(
                        lj_sys.positions,
                        system=lj_sys.system,
                        integrator=openmm.VerletIntegrator(1),
                ))
            for _
            in range(4)
        ]

        mapper = OpenMMProcPoolWorkMapper(
            platform="CPU",
            device_ids=[0,1],
            global_platform_properties={"Threads" : "1"},
        )

        mapper.init()
        new_states = mapper.map(
            [
                OpenMMTask(
                    runner=openmm_runner,
                    segment_length=2,
                )
                for _ in range(len(init_states))
            ],
            init_states,
        )

class TestOpenMMRayPoolWorkMapper:

    @pytest.mark.ray
    def test_all(self, openmm_runner):

        lj_sys = LennardJonesPair()
        
        init_states = [
            OpenMMState(gen_sim_state(
                        lj_sys.positions,
                        system=lj_sys.system,
                        integrator=openmm.VerletIntegrator(1),
                ))
            for _
            in range(4)
        ]

        mapper = OpenMMRayPoolWorkMapper(
            platform="CPU",
            device_ids=[0,1],
            global_platform_properties={"Threads" : "1"},
            # ray_init_args={
                
            # }
        )

        mapper.init()
        new_states = mapper.map(
            [
                OpenMMTask(
                    runner=openmm_runner,
                    segment_length=100000,
                )
                for _ in range(len(init_states))
            ],
            init_states,
        )
        
