# Standard Library
from typing import TypedDict, Annotated

import numpy as np
from numpy.typing import NDArray
import attrs

# First Party Library
from wepy.typing import Shape
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NothingDecisionEnum,
)
from wepy.resampling.resamplers.resampler import Resampler, ResamplerABC
from wepy.walker import Walker
from wepy.util.attrs import AttrsMappingMixin
from wepy.resampling.decisions.no_decision import NoDecisionRecord

@attrs.define
class NoResamplerResamplingRecord(AttrsMappingMixin):
    decision_id: Annotated[
        NDArray[np.int64],
        Shape((1,)),
    ]
    # NOTE,UGLY: It isn't strictly necessary to have multiple target
    # indices for this type of resampling record to have all the
    # information, but all of the downstream infrastucture for
    # interpreting them relies on there being multiple indices so we
    # don't want to break this for this record that is only used for
    # troubleshooting really.
    target_idxs: Annotated[
        NDArray[np.int64],
        Shape((1,1,)),
    ]


@attrs.define
class NoResamplerResamplerRecord(AttrsMappingMixin):
    pass


class NoResampler(ResamplerABC):
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
        list[NoDecision],
        list[NoResamplerResamplerRecord],
    ]:

        # normally decide is only for a single step and so does not
        # include the step_idx, so we add this to the records, and
        # convert the target idxs and decision_id to feature vector
        # arrays
        _resampling_data = []
        for walker_idx in range(len(walkers)):

            # UGLY: we need to wrap the field data into the shape
            walker_record = NoResamplerResamplingRecord(
                decision_id=np.array([NothingDecisionEnum.NOTHING.value]),
                # NOTE: two dimensions to match the target_idxs shape
                target_idxs=np.array([[walker_idx]]),
            )

            _resampling_data.append(walker_record)

        # only a single step of decisions
        resampling_data = _resampling_data

        # there is no change in state in the resampler so there are no
        # resampler records
        resampler_data = [NoResamplerResamplerRecord()]

        # the resampled walkers are just the walkers
        return walkers, resampling_data, resampler_data


@attrs.define
class NoResamplerFactory:

    @classmethod
    def type(cls) -> type[NoResampler]:
        return NoResampler

    def __call__(self, num_cores: int) -> NoResampler:
        return NoResampler()
