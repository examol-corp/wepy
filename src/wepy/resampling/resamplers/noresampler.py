# Standard Library
from typing import TypedDict

# First Party Library
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NothingDecisionEnum,
)
from wepy.resampling.resamplers.resampler import Resampler, ResamplerABC
from wepy.walker import Walker


class NoResamplerResamplingData(TypedDict):
    decision_id: int
    target_idxs: tuple[int, ...]


class NoResamplerResamplerData(TypedDict):
    pass


class NoResampler(Resampler):
    """The resampler which does nothing."""

    DECISION = NoDecision

    # must reset these when you change the decision
    RESAMPLING_FIELDS = DECISION.FIELDS + ResamplerABC.CYCLE_FIELDS
    RESAMPLING_SHAPES = DECISION.SHAPES + ResamplerABC.CYCLE_SHAPES
    RESAMPLING_DTYPES = DECISION.DTYPES + ResamplerABC.CYCLE_DTYPES

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + ResamplerABC.CYCLE_RECORD_FIELDS

    def resample(
        self,
        walkers: list[Walker],
    ) -> tuple[
        list[Walker],
        list[list[NoResamplerResamplingData]],
        list[NoResamplerResamplerData],
    ]:

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
        return walkers, resampling_data, resampler_data


class NoResamplerFactory:

    def __init__(self) -> None:
        pass

    def __call__(self, num_cores: int) -> NoResampler:
        return NoResampler()
