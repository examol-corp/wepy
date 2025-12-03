# from wepy.walker import Walker, WalkerState
from wepy.work_mapper.proc_pool_mapper import ProcPoolMapper
# NOTE: that you must import from a module (as opposed to inline in
# the test file) or you have problems with importing modules when
# using "spawn" process start method
from wepy.runners.mock import MockState, MockError, MockRunner, MockTask

def test_ProcPoolMapper():

    poolmapper = ProcPoolMapper(num_workers=2)

    poolmapper.init()

    results = poolmapper.map(
            [
                MockTask(
                    MockRunner(),
                    segment_length=10,
                    fail=False,
                )
                for _ in range(3)
            ],
            [
                MockState(0),
                MockState(1),
                MockState(2),
            ],
    )

    assert results == [
        MockState(10),
        MockState(11),
        MockState(12),
    ]
