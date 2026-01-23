# Standard Library
from enum import IntEnum
from typing import TypedDict

# Third Party Library
import attrs

# First Party Library
from wepy.resampling.decisions.decision import BaseDecisionABC, BaseDecisionRecord
from wepy.walker import Walker


class NothingDecisionEnum(IntEnum):
    """Enumeration of the decision values for doing nothing."""

    NOTHING = 0
    """Do nothing with the walker."""

class NoDecisionRecordDict(TypedDict):
    decision_id: int
    target_idx: tuple[int, ...]

@attrs.define
class NoDecisionRecord(BaseDecisionRecord):
    decision_id: int = attrs.field()
    target_idxs: tuple[int, ...] = attrs.field()

    @decision_id.validator
    def _check_decision_id(self, attribute, value) -> None:

        if value != NothingDecisionEnum.NOTHING.value:
            raise ValueError(f"Invalid decision_id ({value}) must be {NothingDecisionEnum.NOTHING.value}")

    @target_idxs.validator
    def _check_target_idxs(self, attribute, value) -> None:

        if len(value) < 1:
            raise ValueError(
                f"'target_idxs' must have at least one entry."
            )

        if any(idx < 0 for idx in value):

            raise ValueError(
                f"'target_idxs' values must be non-negative, received: {value}"
            )
        

    def to_dict(self) -> NoDecisionRecordDict:
        return attrs.asdict(self)


class NoDecision(BaseDecisionABC):
    """Decision for a resampling process that does no resampling."""

    ENUM = NothingDecisionEnum
    DEFAULT_DECISION = ENUM.NOTHING
    DECISION_RECORD = NoDecisionRecord

    FIELDS = BaseDecisionABC.FIELDS
    SHAPES = BaseDecisionABC.SHAPES
    DTYPES = BaseDecisionABC.DTYPES

    RECORD_FIELDS = BaseDecisionABC.RECORD_FIELDS

    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,)

    @classmethod
    def action(
        cls,
        walkers: list[Walker],
        decisions: list[NoDecisionRecord],
    ) -> list[Walker]:
        # list for the modified walkers
        mod_walkers: list[Walker] = [None for i in range(len(walkers))]
        # go through each decision and perform the decision
        # instructions
        for step_idx, step_recs in enumerate(decisions):
            for walker_idx, decision_record in enumerate(step_recs):

                if decision_record.decision_id == cls.ENUM.NOTHING.value:

                    target_idx = decision_record.target_idxs[0]
                    
                    # check to make sure a walker doesn't already exist
                    # where you are going to put it
                    if mod_walkers[target_idx] is not None:
                        raise ValueError(
                            f"Multiple walkers assigned to position {target_idx}"
                        )

                    # put the walker in the position specified by the
                    # instruction
                    mod_walkers[target_idx] = walkers[walker_idx]

        return mod_walkers

    @classmethod
    def parents(cls, step: list[NoDecisionRecord]) -> list[int]:

        step_parents = [None for i in range(len(step))]
        for parent_idx, parent_rec in enumerate(step):

            step_parents[parent_rec.target_idxs[0]] = parent_idx

        return step_parents
