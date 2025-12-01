import pytest
import attrs
from wepy.walker import Walker
from wepy.work_mapper.serial import SerialMapper
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.runners.runner import NoRunner
from wepy.runners.mock import MockRunner, MockState, MockError

from wepy.sim_manager import Manager

@pytest.fixture
def sim_components() -> tuple[
        list[Walker],
        NoRunner,
        MockRunner,
        SerialMapper,
]:

    num_walkers = 4

    init_walker_weight = 1 / num_walkers
    init_walkers = [
        Walker(
            state=MockState(1),
            weight=init_walker_weight,
        )
        for walker_state
        in range(num_walkers)
    ]

    return init_walkers, MockRunner(), NoResampler(), SerialMapper()

class TestManager:

    def test___init__(self, sim_components):

        manager = Manager(*sim_components)

        assert len(manager.reporters) == 0
        assert manager.work_mapper == sim_components[3]

    def test_init(self, sim_components):

        manager = Manager(*sim_components)

        manager.init()

    def test_cleanup(self, sim_components):

        manager = Manager(*sim_components)

        manager.cleanup()
        
    def test_run_segment(self, sim_components):

        init_walkers, runner, resampler, mapper = sim_components

        manager = Manager(*sim_components)
        manager.init()

        new_states = manager.run_segment(
            [walker.state for walker in init_walkers],
            1,
            0,
        )

        # test if something fails
        manager = Manager(
            init_walkers,
            MockRunner(fail=True),
            resampler,
            mapper,
        )
        manager.init()

        with pytest.raises(MockError):
            manager.run_segment(
                [walker.state for walker in init_walkers],
                1,
                0,
            )

    def test_run_cycle(self, sim_components):

        init_walkers, runner, resampler, mapper = sim_components

        manager = Manager(*sim_components)
        manager.init()

        new_states = manager.run_cycle(
            init_walkers,
            1,
            0,
        )

        # test if something fails
        manager = Manager(
            init_walkers,
            MockRunner(fail=True),
            resampler,
            mapper,
        )
        manager.init()

        with pytest.raises(MockError):
            manager.run_cycle(
                init_walkers,
                1,
                0,
            )

    def test_run_simulation(self, sim_components):

        manager = Manager(*sim_components)
        manager.init()

        new_walkers, _ = manager.run_simulation(2, 2, num_workers=None)


        new_walkers, _ = manager.run_simulation(
            2,
            2,
            num_workers=None,
            continue_run_idx=0,
        )

    def test_run_simulation_by_time(self, sim_components):

        manager = Manager(*sim_components)
        manager.init()

        new_walkers, _ = manager.run_simulation_by_time(
            1,
            2,
            num_workers=None,
        )

        new_walkers, _ = manager.run_simulation_by_time(
            1,
            2,
            num_workers=None,
            continue_run_idx=0,
        )
