# Standard Library

# Third Party Library
import attrs

# First Party Library
from wepy.work_mapper.serial import SerialMapper

# some minimal definitions for testing a concrete work mapper


@attrs.define
class RizzWalkerState:
    rizz: int


def rizz_run(walker_state: RizzWalkerState, segment_length: int) -> RizzWalkerState:

    return attrs.evolve(
        walker_state,
        rizz=(walker_state.rizz + segment_length),
    )


@attrs.define
class RizzTask:

    multiple: int

    def __call__(self, state: RizzWalkerState, segment_length: int) -> RizzWalkerState:

        return rizz_run(state, segment_length=segment_length, multiple=self.multiple)


def test_rizz_walker():
    assert rizz_run(
        RizzWalkerState(rizz=1),
        1,
    ) == RizzWalkerState(rizz=2)


class TestMapper:

    def test_map(self):

        mapper = SerialMapper()

        mapper.init()

        assert mapper.map(
            rizz_run,
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
            [1, 2, 2],
        ) == [
            RizzWalkerState(2),
            RizzWalkerState(3),
            RizzWalkerState(4),
        ]

        assert len(mapper.get_worker_segment_times()[0]) == 3
