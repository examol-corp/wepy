import functools
import attrs
from wepy.walker import Walker, WalkerState
from wepy.interface import (
    WorkMapperFactoryArgs,
)
from wepy.work_mapper.serial import SerialMapper

# some minimal definitions for testing a concrete work mapper

@attrs.define
class RizzWalkerState:
    rizz: int


def rizz_run(walker_state: RizzWalkerState, delta: int, multiple: int) -> RizzWalkerState:

    return attrs.evolve(
        walker_state,
        rizz=(walker_state.rizz + delta) * multiple
    )

@attrs.define
class RizzTask:

    delta: int
    multiple: int

    def __call__(self, state: RizzWalkerState) -> RizzWalkerState:

        return rizz_run(state, delta=self.delta, multiple=self.multiple)

def test_rizz_walker():
    assert rizz_run(
        RizzWalkerState(rizz=1),
        1,
        2,
    ) == RizzWalkerState(rizz=4)


class TestMapper:

    def test_map(self):

        mapper = SerialMapper(WorkMapperFactoryArgs(num_workers=0))

        mapper.init()

        assert mapper.map(
            [
                RizzTask(
                    *args,
                )
                for args
                in zip(
                [1, 2, 2],
                [2, 2, 2],
                )
            ],
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
        ) == [
            RizzWalkerState(4),
            RizzWalkerState(6),
            RizzWalkerState(8),
        ]

        assert len(mapper.get_worker_segment_times()[0]) == 3
