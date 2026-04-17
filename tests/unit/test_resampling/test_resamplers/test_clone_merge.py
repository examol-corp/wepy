# Third Party Library
import pytest

# First Party Library
from wepy.resampling.decisions.clone_merge import (
    CloneMergeDecisionRecord,
)
from wepy.resampling.resamplers.clone_merge import CloneMergeResampler
from wepy.resampling.resamplers.resampler import ResamplerError
from wepy.runners.mock import MockState
from wepy.walker import Walker

# class MockCloneMergeResampler(ResamplerABC):


class Test_CloneMergeResampler:

    def test___init__(self):

        CloneMergeResampler()

        with pytest.raises(ResamplerError):
            CloneMergeResampler(
                pmin=0.1,
                pmax=0.01,
            )

        with pytest.raises(ResamplerError):
            CloneMergeResampler(
                pmin=1.0,
                pmax=0.01,
            )

        with pytest.raises(ResamplerError):
            CloneMergeResampler(
                pmin=0.1,
                pmax=1.0,
            )

    def test__init_walker_actions(self):

        assert CloneMergeResampler()._init_walker_actions(4) == [
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(0,)),
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(1,)),
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(2,)),
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(3,)),
        ]

    def test__check_resampled_walkers(self):

        CloneMergeResampler(
            pmin=0.1,
            pmax=0.4,
        )._check_resampled_walkers(
            [
                Walker(
                    MockState(1),
                    weight=0.1,
                ),
                Walker(
                    MockState(1),
                    weight=0.4,
                ),
            ]
        )

        with pytest.raises(ResamplerError):
            CloneMergeResampler(
                pmin=0.1,
                pmax=0.4,
            )._check_resampled_walkers(
                [
                    Walker(
                        MockState(1),
                        weight=0.01,
                    ),
                    Walker(
                        MockState(1),
                        weight=0.4,
                    ),
                ]
            )

        with pytest.raises(ResamplerError):
            CloneMergeResampler(
                pmin=0.1,
                pmax=0.4,
            )._check_resampled_walkers(
                [
                    Walker(
                        MockState(1),
                        weight=0.1,
                    ),
                    Walker(
                        MockState(1),
                        weight=0.5,
                    ),
                ]
            )

    def test_assign_clones(self):

        resampler = CloneMergeResampler(
            pmin=0.1,
            pmax=0.4,
        )

        with pytest.raises(ResamplerError):
            resampler.assign_clones(
                merge_groups=[
                    [],
                    [],
                ],
                walker_clone_nums=[0, 0, 0],
            )

        assert resampler.assign_clones(
            merge_groups=[[], []],
            walker_clone_nums=[0, 0],
        ) == [
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(0,)),
            CloneMergeDecisionRecord(decision_id=1, target_idxs=(1,)),
        ]

        assert resampler.assign_clones(
            merge_groups=[
                [],
                [2],
                [],
            ],
            walker_clone_nums=[1, 0, 0],
        ) == [
            CloneMergeDecisionRecord(decision_id=2, target_idxs=(0, 2)),
            CloneMergeDecisionRecord(decision_id=4, target_idxs=(1,)),
            CloneMergeDecisionRecord(decision_id=3, target_idxs=(1,)),
        ]

        with pytest.raises(ResamplerError):

            resampler.assign_clones(
                merge_groups=[
                    [2],
                    [],
                    [],
                ],
                walker_clone_nums=[1, 0, 0],
            )
