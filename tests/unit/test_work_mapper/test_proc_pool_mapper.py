import attrs
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.proc_pool_mapper import ProcPoolMapper

# some minimal definitions for testing a concrete work mapper

@attrs.define
class RizzWalkerState:
    rizz: int

def rizz_run(walker_state: RizzWalkerState, delta: int, multiple: int) -> RizzWalkerState:

    return attrs.evolve(
        walker_state,
        rizz=(walker_state.rizz + delta) * multiple
    )

def test_rizz_walker():
    assert rizz_run(
        RizzWalkerState(rizz=1),
        1,
        2,
    ) == RizzWalkerState(rizz=4)

def test_ProcPoolMapper():

    poolmapper = ProcPoolMapper()

    poolmapper.init(rizz_run, num_workers=1)

    results = poolmapper.map(
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
            [1, 2, 2],
            [2, 2, 2],
    )

    assert results == [
            RizzWalkerState(4),
            RizzWalkerState(6),
            RizzWalkerState(8),
        ]


    poolmapper = ProcPoolMapper()

    poolmapper.init(rizz_run, num_workers=2)

    results = poolmapper.map(
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
            [1, 2, 2],
            [2, 2, 2],
    )

    assert results == [
            RizzWalkerState(4),
            RizzWalkerState(6),
            RizzWalkerState(8),
        ]

    poolmapper = ProcPoolMapper()

    poolmapper.init(rizz_run, num_workers=3)

    results = poolmapper.map(
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
            [1, 2, 2],
            [2, 2, 2],
    )

    assert results == [
            RizzWalkerState(4),
            RizzWalkerState(6),
            RizzWalkerState(8),
        ]

    poolmapper = ProcPoolMapper()

    poolmapper.init(rizz_run, num_workers=3, proc_start_method="fork")

    results = poolmapper.map(
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(2),
            ],
            [1, 2, 2],
            [2, 2, 2],
    )

    assert results == [
            RizzWalkerState(4),
            RizzWalkerState(6),
            RizzWalkerState(8),
        ]

    # use worker args to fill in one of the args
    poolmapper = ProcPoolMapper()

    poolmapper.init(
        rizz_run,
        num_workers=2,
        worker_args=[
            {"multiple" : 2},
            {"multiple" : 3},
        ])

    results = poolmapper.map(
            [
                RizzWalkerState(1),
                RizzWalkerState(1),
                RizzWalkerState(1),
            ],
            [1, 1, 1],
    )
