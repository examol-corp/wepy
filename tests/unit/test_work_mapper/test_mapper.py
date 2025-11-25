# Standard Library
import logging

logger = logging.getLogger(__name__)
# Standard Library
import time

# Third Party Library
import pytest

# First Party Library
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.mapper import (
    ABCMapper,
    Mapper,
    TaskException,
    Task,
    WrapperException,
    TaskException,
    ABCWorkerMapper,
    WorkerException,
    WorkerKilledError,
    WorkerMapper,
    Worker,
)
from wepy.work_mapper.task_mapper import (
    TaskMapper,
    WalkerTaskProcess,
)
from wepy.work_mapper.worker import Worker, WorkerMapper

ARGS = (0, 1, 2)


def gen_walkers():
    return [Walker(WalkerState(**{"num": arg}), 1 / len(ARGS)) for arg in ARGS]


# test basic functionality
def task_pass(walker):
    # simulate it actually taking some time
    n = walker.state["num"]
    return Walker(WalkerState(**{"num": n + 1}), walker.weight)


TASK_PASS_ANSWER = [n + 1 for n in ARGS]

class TestABCMapper:
    def test___init__(self):

        no_mapper = ABCMapper()
        assert no_mapper.segment_func is None
        assert no_mapper.attributes == {}

        mapper = ABCMapper(
            segment_func=(lambda x: )
        )
        assert no_mapper.segment_func is None
        assert no_mapper.attributes == {}
        

    def test_init(self):
        assert False

    def test_cleanup(self):
        assert False
        
class TestMapper:
    def test___init__(self):
        assert False

    def test_map(self):
        assert False
    def test_worker_segment_times(self):
        assert False
        

class TestWorkMappers:
    def test_mapper(self):
        mapper = Mapper(segment_func=task_pass)

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

    def test_worker_mapper(self):
        mapper = WorkerMapper(segment_func=task_pass, num_workers=3, worker_type=Worker)

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

    def test_task_mapper(self):
        mapper = TaskMapper(
            segment_func=task_pass, num_workers=3, walker_task_type=WalkerTaskProcess
        )

        mapper.init()

        results = mapper.map(gen_walkers())

        assert all(
            [res.state["num"] == TASK_PASS_ANSWER[i] for i, res in enumerate(results)]
        )

        mapper.cleanup()

        time.sleep(1)

class TestTask:

    def test___init__(self):
        assert False

    def test___call__(self):
        assert False

class TestWrapperException:

    def test___init__(self):
        assert False


class TestABCWorkerMapper:

    def test___init__(self):
        assert False

    def test_init(self):
        assert False

    def test_cleanup(self):
        assert False

    def test__make_task(self):
        assert False

class TestWorkerMapper:

    def test___init__(self):
        assert False
        
    def test_init(self):
        assert False

    def test__sigterm_shutdown(self):
        assert False

    def test_force_shutdown(self):
        assert False

    def test_cleanup(self):
        assert False

    def test_map(self):
        assert False

class TestWorker:

    def test___init__(self):
        assert False

    def test_run(self):
        assert False

    def test__sigterm_shutdown(self):
        assert False

    def test__shutdown(self):
        assert False

    def test__run_worker(self):
        assert False

    def test_run_task(self):
        assert False

    def test__run_task(self):
        assert False
        
# test that task failures are passed up properly
def task_fail(walker):
    n = walker.state["num"]
    if n == 1:
        raise ValueError("No soup for you!!")
    else:
        return Walker(WalkerState(**{"num": n + 1}), walker.weight)


class TestTaskFail:
    ARGS = ((0, 1, 2),)

    def test_mapper(self):
        mapper = Mapper(segment_func=task_fail)

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()

    def test_worker_mapper(self):
        mapper = WorkerMapper(segment_func=task_fail, num_workers=3, worker_type=Worker)

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()

    def test_task_mapper(self):
        mapper = TaskMapper(
            segment_func=task_fail, num_workers=3, walker_task_type=WalkerTaskProcess
        )

        mapper.init()

        with pytest.raises(TaskException) as task_exc_info:
            results = mapper.map(gen_walkers())

        mapper.cleanup()
