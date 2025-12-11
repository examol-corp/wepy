# Standard Library
import contextlib
import copy
import logging
import logging.config
import logging.handlers
import multiprocessing as mp
import os
from typing import Generator

logger = logging.getLogger(__name__)

_BASE_WORKER_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": None,  # injected at runtime
        }
    },
    "root": {
        "handlers": ["queue"],
        "level": "NOTSET",  # defer filtering to parent
    },
}


def proc_pool_worker_setup(log_queue: mp.Queue) -> None:
    """Pool(initializer=) function that handles logging properly.

    Does its best to inherit log settings from a parent
    process. Requires a logging queue that the log messages are sent
    on.

    Works properly with 'spawn' start method.



    """

    config = copy.deepcopy(_BASE_WORKER_LOGGING_CONFIG)

    config["handlers"]["queue"]["queue"] = log_queue
    parent_loggers = logging.root.manager.loggerDict
    for name, logger in parent_loggers.items():
        if isinstance(logger, logging.Logger):
            config.setdefault("loggers", {})[name] = {
                "level": logging.getLevelName(logger.level),
                "propagate": logger.propagate,
                "handlers": [],
                "filters": [f.__class__.__name__ for f in logger.filters],
            }

    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.info("Configured logging in worker process")


def _dummy_task(foo: int) -> int:
    """Just a dummy function used for testing.

    For 'spawn' we need to have it importable thus it is defined here
    and not in a test.

    """

    logger = logging.getLogger("dummy-task")
    logger.info("Executing dummy task")

    return foo + 1


class WorkerFormatter(logging.Formatter):
    def __init__(self, base_formatter: logging.Formatter):
        self.base_formatter = base_formatter

    def format(self, record):
        # Ensure process info exists
        if not hasattr(record, "processName"):
            record.processName = mp.current_process().name
        if not hasattr(record, "process"):
            record.process = mp.current_process().pid

        # Prepend process info to the actual message

        # the formatted msg string
        original_msg = record.getMessage()
        record.msg = f"[{record.processName} | PID {record.process}] {original_msg}"
        # ensure no old args are re-applied
        record.args = ()

        # Use the base formatter for the rest
        formatted = self.base_formatter.format(record)

        # Restore original message so we don't mutate it permanently
        record.msg = original_msg
        return formatted


@contextlib.contextmanager
def queue_listener_context(mp_ctx) -> Generator[None, None, None]:

    logger.info("Setting up queue logging infrastructure")
    root_logger = logging.getLogger()
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):

        record = old_factory(*args, **kwargs)
        if not hasattr(record, "processName"):
            record.processName = mp.current_process().name

        if not hasattr(record, "process"):
            record.process = os.getpid()

    old_formatters = [handler.formatter for handler in root_logger.handlers]

    listener_handlers = []
    for handler in root_logger.handlers:
        new_handler = copy.copy(handler)
        new_handler.setFormatter(WorkerFormatter(handler.formatter))
        listener_handlers.append(new_handler)

    logger.info("Starting Queue")
    log_queue = mp_ctx.Queue()
    listener = logging.handlers.QueueListener(log_queue, *listener_handlers)
    logger.info("Starting QueueListener")
    listener.start()
    logger.info("Listener started")

    try:
        yield log_queue
    finally:
        logger.info("Shutting down log listener resources")

        logger.info("Stopping listener")
        listener.stop()
        logger.info("Listener stopped")

        for handler, formatter in zip(
            root_logger.handlers, old_formatters, strict=True
        ):
            handler.setFormatter(formatter)

        logging.setLogRecordFactory(old_factory)

        logger.info("Closing logging Queue")
        try:
            log_queue.close()
            log_queue.join_thread()
        except Exception as exc:
            logger.error(f"Exception in log Queue closing, continuing: {exc}")
            pass
        except:
            logger.info("Logger Queue shut down cleanly")

        logger.info("Finished context cleanup")
