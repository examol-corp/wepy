"""Reference implementation of serial work mapper."""

from typing import Callable, Literal, Generic, TypeVar, Protocol, Any
import logging

# Standard Library
import multiprocessing as mp
import queue as pyq
import signal
import sys
import time
import traceback
from warnings import warn

from wepy.walker import Walker
from wepy.work_mapper.base import WorkMapper, AnyWalker

logger = logging.getLogger(__name__)

class ABCWorkerMapper(ABCMapper):
    def __init__(
        self,
        num_workers: int | None = None,
        segment_func: SegmentFunc = None,
        proc_start_method: Literal["fork", "spawn", "forkserver"] = "fork",
        **kwargs,
    ) -> None:
        """Constructor for WorkerMapper.

        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn.

        segment_func : callable, optional
            Set a default segment_func. Typically set at runtime.

        proc_start_method : str or None
            A string indicating the type of process start method to
            use from python multiprocessing typically 'fork', 'spawn',
            or 'forkserver', or the platform default for None. See
            documentation. Generates a context with the method
            multiprocessing.get_context(proc_start_method) on `init`.

        """

        super().__init__(segment_func=segment_func, **kwargs)

        self._proc_start_method = proc_start_method

        self._num_workers = num_workers
        self._worker_segment_times = None

        if num_workers is not None:
            self._worker_segment_times = {i: [] for i in range(self.num_workers)}

    def init(self, num_workers=None, segment_func=None, **kwargs):
        """Runtime initialization and setting of function to map over walkers.

        Parameters
        ----------
        num_workers : int
            The number of worker processes to spawn

        segment_func : callable implementing the Runner.run_segment interface

        """

        super().init(segment_func=segment_func)

        # create the multiprocessing context to use for spawning
        # processes here
        self._mp_ctx = mp.get_context(method=self._proc_start_method)

        # the number of workers must be given here or set as an object attribute
        if num_workers is None and self.num_workers is None:
            raise ValueError(
                "The number of workers must be given, received {}".format(num_workers)
            )

        # if the number of walkers was given for this init() call use
        # that, otherwise we use the default that was specified when
        # the object was created
        elif num_workers is not None and self.num_workers is None:
            self._num_workers = num_workers

        # update the worker segment times
        self._worker_segment_times = {i: [] for i in range(self.num_workers)}

    def cleanup(self, **kwargs):
        # ALERT: is this all we need to do? I have a hunch there is
        # more caveats, but these context objects are not really
        # documented

        # make sure the context for this work mapper is destroyed
        del self._mp_ctx

    @property
    def num_workers(self):
        """The number of worker processes."""
        return self._num_workers

    @property
    def worker_segment_times(self):
        """The run timings for each segment for each walker.

        Returns
        -------
        worker_seg_times : dict of int : list of float
            Dictionary mapping worker indices to a list of times in
            seconds for each segment run.

        """
        return self._worker_segment_times

    def _make_task(self, *args, **kwargs):
        """Generate a task from 'segment_func' attribute.

        Similar to partial evaluation (or currying).

        Args will be eventually used as the arguments to the call of
        'segment_func' by the worker processes when they receive the
        task from the queue.

        Returns
        -------
        task : Task object

        """
        return Task(self._func, *args, **kwargs)
