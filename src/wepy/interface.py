from typing import Generic, TypeVar, Protocol

import attrs

from wepy.walker import WalkerState

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)

@attrs.define
class WorkMapperFactoryArgs:
    num_workers: int

@attrs.define
class RunnerGenTaskArgs(Generic[WalkerState_]):
    segment_length: int
    cycle_idx: int
    states: list[WalkerState_]

class Task(Protocol[WalkerState_]):

    def __call__(
            self,
            walker_state: WalkerState_,
    ) -> WalkerState_:
        ...
