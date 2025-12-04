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

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)

class Runner(Protocol[WalkerState_]):
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

    def run_segment(
        self,
        state: WalkerState_,
        segment_length: int | float,
    ) -> WalkerState_:
        return state
