# Standard Library
from enum import IntEnum

# Third Party Library
import attrs

# First Party Library
from wepy.resampling.decisions.decision import BaseDecisionABC, BaseDecisionRecord
from wepy.walker import Walker


class NothingDecisionEnum(IntEnum):
    """Enumeration of the decision values for doing nothing."""

    NOTHING = 0
    """Do nothing with the walker."""


@attrs.define
class NoDecisionRecord(BaseDecisionRecord):
    decision_id: int
    target_idx: int


class NoDecision(BaseDecisionABC):
    """Decision for a resampling process that does no resampling."""

    ENUM = NothingDecisionEnum
    DEFAULT_DECISION = ENUM.NOTHING

    FIELDS = BaseDecisionABC.FIELDS + ("target_idxs",)
    SHAPES = BaseDecisionABC.SHAPES + (Ellipsis,)
    DTYPES = BaseDecisionABC.DTYPES + (int,)

    RECORD_FIELDS = BaseDecisionABC.RECORD_FIELDS + ("target_idxs",)

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
            for walker_idx, decision in enumerate(step_recs):

                if decision.decision_id == cls.ENUM.NOTHING.value:
                    # check to make sure a walker doesn't already exist
                    # where you are going to put it
                    if mod_walkers[decision.target_idx] is not None:
                        raise ValueError(
                            "Multiple walkers assigned to position {}".format(
                                decision.target_idx
                            )
                        )

                    # put the walker in the position specified by the
                    # instruction
                    mod_walkers[decision.target_idx] = walkers[walker_idx]

        return mod_walkers

    @classmethod
    def parents(cls, step: list[NoDecisionRecord]) -> list[int]:

        step_parents = [None for i in range(len(step))]
        for parent_idx, parent_rec in enumerate(step):

            step_parents[parent_rec.target_idx] = parent_idx

        return step_parents
