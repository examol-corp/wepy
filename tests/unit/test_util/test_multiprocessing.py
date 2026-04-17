# Standard Library
import logging
import multiprocessing as mp

# First Party Library
from wepy.util.multiprocessing import _dummy_task, proc_pool_worker_setup


def test__dummy_task():

    assert _dummy_task(1) == 2


def test_proc_pool_worker_setup(caplog):

    mp_ctx = mp.get_context(method="spawn")

    log_queue = mp_ctx.Queue()
    handlers = list(logging.getLogger().handlers)
    listener = logging.handlers.QueueListener(log_queue, *handlers)
    listener.start()

    with mp_ctx.Pool(
        processes=1,
        initializer=proc_pool_worker_setup,
        initargs=(log_queue,),
    ) as pool:

        # NOTE: this is the hardcoded logger in the dummy function
        with caplog.at_level(logging.INFO, logger="dummy-task"):
            result = pool.apply(_dummy_task, (1,))

    assert result == 2

    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[0].msg == "Configured logging in worker process"
    assert caplog.records[1].levelname == "INFO"
    assert caplog.records[1].msg == "Executing dummy task"
