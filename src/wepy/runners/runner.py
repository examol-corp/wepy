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
import logging
from enum import IntEnum
from typing import Callable, Literal, Protocol, TypeVar

# Third Party Library
import attrs
from immutables import Map as frozenmap

# First Party Library
from wepy.walker import WalkerState

logger = logging.getLogger(__name__)


@attrs.define
class RunSegmentData:
    segment_split_time: float


WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
RunSegmentData_ = TypeVar("RunSegmentData_", bound=RunSegmentData)


class RunnerStatus(IntEnum):
    PRE_INITIALIZATION = 0
    INITIALIZED = 1
    PRE_CYCLE = 2
    POST_SEGMENT = 3
    POST_CYCLE = 4


class RunnerEvent(IntEnum):
    INIT = 0
    PRE_CYCLE = 1
    POST_SEGMENT = 2
    POST_CYCLE = 3


class RunnerStateMachineError(Exception):
    pass


class RunnerStateTransitionError(RunnerStateMachineError):
    """Indicates an error with the runner state machine transition."""

    pass


class RunnerStateError(RunnerStateMachineError):
    """Indicates an error relating to the current state of the runner."""

    pass


# State machine table that defines what are the valid states to
# transition to another state. None for the initial states
RUNNER_STATE_TRANSITION_TABLE: frozenmap[
    RunnerStatus,
    frozenmap[RunnerEvent, RunnerStatus],
] = frozenmap(
    {
        RunnerStatus.PRE_INITIALIZATION: frozenmap(
            {
                RunnerEvent.INIT: RunnerStatus.INITIALIZED,
            }
        ),
        RunnerStatus.INITIALIZED: frozenmap(
            {
                RunnerEvent.PRE_CYCLE: RunnerStatus.PRE_CYCLE,
            }
        ),
        RunnerStatus.PRE_CYCLE: frozenmap(
            {
                RunnerEvent.POST_SEGMENT: RunnerStatus.POST_SEGMENT,
            }
        ),
        RunnerStatus.POST_SEGMENT: frozenmap(
            {
                RunnerEvent.POST_CYCLE: RunnerStatus.POST_CYCLE,
            }
        ),
        RunnerStatus.POST_CYCLE: frozenmap(
            {
                RunnerEvent.PRE_CYCLE: RunnerStatus.PRE_CYCLE,
            }
        ),
    }
)


@attrs.define
class RunnerStateMachine:
    state: RunnerStatus = attrs.field(
        default=RunnerStatus.PRE_INITIALIZATION,
    )

    def validate_event(self, event: RunnerEvent) -> Literal[True]:
        state_transitions = RUNNER_STATE_TRANSITION_TABLE[self.state]
        if event not in state_transitions:
            raise RunnerStateTransitionError(
                f"Runner is in state {self.state.name}:{self.state.value},"
                f"event {event.name}:{event.value} is not a valid."
                f" Choose from: {set(state_transitions.keys())}"
            )
        else:
            return True

    def send(self, event: RunnerEvent) -> RunnerStatus:

        logger.info(f"Received event: {event.name}:{event.value}")
        self.validate_event(event)

        state_transitions = RUNNER_STATE_TRANSITION_TABLE[self.state]
        new_state = state_transitions[event]

        logger.info(
            "Transitioning runner state:"
            f" {self.state.name}:{self.state.value} -> {new_state.name}:{new_state.value}"
        )
        self.state = new_state

        return self.state


class Runner(Protocol[WalkerState_, RunSegmentData_]):
    """Abstract base class for the Runner interface."""

    @property
    def status(self) -> RunnerStatus: ...

    def init(self) -> None: ...

    def pre_cycle(self) -> None:
        """Perform pre-cycle behavior."""
        ...

    def run_segment(
        self,
        walker: WalkerState_,
        segment_length: int,
    ) -> tuple[WalkerState_, RunSegmentData_ | None]:
        """Run dynamics for the walker.

        Parameters
        ----------
        walker : The walker for which dynamics will be propagated.
        segment_length : The numerical value that specifies how much dynamics are to be run.


        Returns
        -------
        new_walker : Walker after dynamics was run, only the state should be modified.
        run_segment_data: Arbitrary data type that is used internally
            in the runner and managers for runner specific data,
            e.g. segment performance metrics.

        """
        ...

    def post_cycle(
        self,
        segments_data: list[RunSegmentData_] | None,
    ) -> None:
        """Perform post-cycle behavior."""
        ...


RunnerFactory = Callable[
    [],
    Runner,
]


@attrs.define
class NoRunner(Runner):
    """Stub Runner that just returns the walkers back with the same state.

    May be useful for testing.
    """

    state_machine: RunnerStateMachine = attrs.field(
        default=attrs.Factory(
            RunnerStateMachine,
        )
    )

    @property
    def status(self) -> RunnerStatus:
        return self.state_machine.state

    def init(self) -> None:
        self.state_machine.send(RunnerEvent.INIT)

    def pre_cycle(self) -> None:
        self.state_machine.send(RunnerEvent.PRE_CYCLE)

    def run_segment(
        self,
        state: WalkerState_,
        segment_length: int | float,
    ) -> tuple[WalkerState_, None]:

        if self.status != RunnerStatus.PRE_CYCLE:
            raise RunnerStateError(
                f"Cannot run a segment in state ({self.status.name}:{self.status.value})"
            )

        return state, None

    def post_cycle(self, segments_data: None) -> None:

        self.state_machine.send(RunnerEvent.POST_SEGMENT)
        self.state_machine.send(RunnerEvent.POST_CYCLE)
