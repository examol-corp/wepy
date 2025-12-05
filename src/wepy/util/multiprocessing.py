import copy
import logging
import logging.handlers
import logging.config
import multiprocessing as mp

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
                "level" : logging.getLevelName(logger.level),
                "propagate" : logger.propagate,
                "handlers" : [],
                "filters" : [
                    f.__class__.__name__
                    for f
                    in logger.filters
                ],
            }

    logging.config.dictConfig(config)

    logger.info("Configured logging in worker process")


def _dummy_task(foo: int) -> int:
    """Just a dummy function used for testing.

    For 'spawn' we need to have it importable thus it is defined here
    and not in a test.

    """

    logger = logging.getLogger("dummy-task")
    logger.info("Executing dummy task")

    return foo + 1
