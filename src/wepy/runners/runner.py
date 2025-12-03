"""Abstract Base classes implementing the Runner interface.

Runner Interface
----------------

All a runner needs to implement is the 'run_segment' method which
should accept a walker and a spec for the length of the segment to run
(e.g. number of dynamics steps).

Additionally, any number of optional key word arguments should be given.

As a matter of convention, classes accessory to a runner (such as
State, Walker, Worker, etc.) should also be put in the same module as
the runner.

See the openmm.py module for an example.

"""

# Standard Library
from typing import Protocol, Any, TypedDict, TypeVar, ParamSpec
import attrs
from wepy.walker import Walker, WalkerState
from wepy.interface import Task, RunnerGenTaskArgs

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)

Task_ = TypeVar("Task_", bound=Task)

class Runner(Protocol[WalkerState_, Task_]):
    """Abstract base class for the Runner interface."""

    def pre_cycle(self) -> None:
        """Perform pre-cycle behavior. run_segment will be called for each
        walker so this allows you to perform changes of state on a
        per-cycle basis.

        Parameters
        ----------
        kwargs : key-word arguments
            Key-value pairs to be interpreted by each runner implementation.

        """
        ...

    def post_cycle(self) -> None:
        """Perform post-cycle behavior. run_segment will be called for each
        walker so this allows you to perform changes of state on a
        per-cycle basis.

        Parameters
        ----------
        kwargs : key-word arguments
            Key-value pairs to be interpreted by each runner implementation.

        """

        ...

    def gen_tasks(self, segment_spec: RunnerGenTaskArgs[WalkerState_]) -> list[Task_]:
        ...

    def run_segment(
        self,
        walker: WalkerState_,
        segment_length: int,
    ) -> WalkerState_:
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : The walker for which dynamics will be propagated.
        segment_length : The numerical value that specifies how much dynamics are to be run.


        Returns
        -------
        new_walker : object implementing the Walker interface
            Walker after dynamics was run, only the state should be modified.

        """
        ...

    def get_last_cycle_segments_split_times(self) -> list[dict[str, float]] | None: ...

@attrs.define
class IdentityTask(Task[WalkerState_]):

    runner: "NoRunner"

    def __call__(self, walker_state: WalkerState_) -> WalkerState_:
        return self.runner.run_segment(walker_state)

@attrs.define
class NoRunner(Runner):
    """Stub Runner that just returns the walkers back with the same state.

    May be useful for testing.
    """

    def pre_cycle(self) -> None:
        pass
    def post_cycle(self) -> None:
        pass
    def get_last_cycle_segments_split_times(self) -> None:
        return None

    def gen_tasks(self, segment_spec: RunnerGenTaskArgs[WalkerState_]) -> list[IdentityTask[WalkerState_]]:

        return [
            IdentityTask(self)
            for state
            in segment_spec.states
        ]
        
    def run_segment(
        self,
        state: WalkerState_,
        segment_length: int | float,
    ) -> WalkerState_:
        return state
