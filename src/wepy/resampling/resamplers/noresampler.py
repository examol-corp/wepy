from typing import TypedDict
import numpy as np
from wepy.walker import Walker
from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NoDecisionRecord,
    NothingDecisionEnum,
)


class NoResamplerResamplingData(TypedDict):
    decision_id: int
    target_idxs: tuple[int, ...]


class NoResamplerResamplerData(TypedDict):
    pass


class NoResampler(Resampler):
    """The resampler which does nothing."""

    DECISION = NoDecision

    # must reset these when you change the decision
    RESAMPLING_FIELDS = DECISION.FIELDS + Resampler.CYCLE_FIELDS
    RESAMPLING_SHAPES = DECISION.SHAPES + Resampler.CYCLE_SHAPES
    RESAMPLING_DTYPES = DECISION.DTYPES + Resampler.CYCLE_DTYPES

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + Resampler.CYCLE_RECORD_FIELDS

    def resample(
        self,
        walkers: list[Walker],
    ) -> tuple[
        list[Walker],
        list[list[NoResamplerResamplingData]],
        list[NoResamplerResamplerData],
    ]:

        # TODO,REFACT: do we really need this
        self._resample_init(walkers=walkers)

        # normally decide is only for a single step and so does not
        # include the step_idx, so we add this to the records, and
        # convert the target idxs and decision_id to feature vector
        # arrays
        _resampling_data: list[NoResamplerResamplingData] = []
        for walker_idx in range(len(walkers)):

            walker_record = NoResamplerResamplingData(
                decision_id=NothingDecisionEnum.NOTHING.value,
                target_idxs=[walker_idx],
            )

            _resampling_data.append(walker_record)

        # only a single step of decisions
        resampling_data = [_resampling_data]

        # there is no change in state in the resampler so there are no
        # resampler records
        resampler_data: list[NoResamplerResamplerData] = [{}]

        # the resampled walkers are just the walkers

        # TODO,REFACT: do we really need this
        self._resample_cleanup(
            resampling_data=resampling_data,
            resampler_data=resampler_data,
            walkers=walkers,
        )

        return walkers, resampling_data, resampler_data
