# Third Party Library
import attrs
import pytest

# First Party Library
from wepy.resampling.decisions.clone_merge import (
    CloneMergeDecisionEnum,
    CloneMergeDecisionRecord,
    MultiCloneMergeDecision,
    CloneMergeDecisionError,
)
from wepy.runners.mock import MockState
from wepy.walker import Walker


class Test_CloneMergeDecisionRecord:

    def test___init__(self):

        # single target records
        CloneMergeDecisionRecord(
            decision_id=1,
            target_idxs=(0,),
        )
        CloneMergeDecisionRecord(
            decision_id=3,
            target_idxs=(0,),
        )
        CloneMergeDecisionRecord(
            decision_id=4,
            target_idxs=(0,),
        )

        # clone
        CloneMergeDecisionRecord(
            decision_id=2,
            target_idxs=(0,1),
        )

        with pytest.raises(ValueError):
            CloneMergeDecisionRecord(
                decision_id=7,
                target_idxs=(0,),
            )

        with pytest.raises(ValueError):
            CloneMergeDecisionRecord(
                decision_id=1,
                target_idxs=(),
            )

        with pytest.raises(ValueError):
            CloneMergeDecisionRecord(
                decision_id=1,
                target_idxs=(-1,),
            )

        with pytest.raises(CloneMergeDecisionError):
            CloneMergeDecisionRecord(
                decision_id=1,
                target_idxs=(0,1,),
            )

        with pytest.raises(CloneMergeDecisionError):
            CloneMergeDecisionRecord(
                decision_id=3,
                target_idxs=(0,1,),
            )
        with pytest.raises(CloneMergeDecisionError):
            CloneMergeDecisionRecord(
                decision_id=4,
                target_idxs=(0,1,),
            )

        with pytest.raises(CloneMergeDecisionError):
            CloneMergeDecisionRecord(
                decision_id=2,
                target_idxs=(0,),
            )

    def test_to_dict(self):

        assert CloneMergeDecisionRecord(
            decision_id=1,
            target_idxs=(0,)
        ).to_dict() == {
            "decision_id" : 1,
            "target_idxs" : (0,),
        }

class TestMultiCloneMergeDecision:

    def test_action(self):
        walker_1 = Walker(
            state=MockState(a=1),
            weight=1.0,
        )

        walker_2 = Walker(
            state=MockState(a=2),
            weight=1.0,
        )
        walker_3 = Walker(
            state=MockState(a=3),
            weight=1.0,
        )

        walkers = [
            walker_1,
            walker_2,
        ]

        # unknown decision number
        with pytest.raises(ValueError):
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": 1000,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [1],
                            }
                        ),
                    ]
                ],
            )

        assert (
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [1],
                            }
                        ),
                    ]
                ],
            )
            == walkers
        )

        # reorder
        assert MultiCloneMergeDecision.action(
            walkers,
            [
                [
                    CloneMergeDecisionRecord(
                        **{
                            "decision_id": CloneMergeDecisionEnum.NOTHING,
                            "target_idxs": [1],
                        }
                    ),
                    CloneMergeDecisionRecord(
                        **{
                            "decision_id": CloneMergeDecisionEnum.NOTHING,
                            "target_idxs": [0],
                        }
                    ),
                ]
            ],
        ) == [walker_2, walker_1]

        # multiple assignment to same slot
        with pytest.raises(ValueError):
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [0],
                            }
                        ),
                    ]
                ],
            )

        # TODO: this should be a more explicit error
        #
        # squashing without filling a slot is an error
        with pytest.raises(KeyError):
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.SQUASH,
                                "target_idxs": [1],
                            }
                        ),
                    ]
                ],
            )
        with pytest.raises(KeyError):
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.NOTHING,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.SQUASH,
                                "target_idxs": [0],
                            }
                        ),
                    ]
                ],
            )

        # provide a keep merge target, but leave a slot open...
        with pytest.raises(ValueError):
            MultiCloneMergeDecision.action(
                walkers,
                [
                    [
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.KEEP_MERGE,
                                "target_idxs": [0],
                            }
                        ),
                        CloneMergeDecisionRecord(
                            **{
                                "decision_id": CloneMergeDecisionEnum.SQUASH,
                                "target_idxs": [0],
                            }
                        ),
                    ]
                ],
            )

        assert MultiCloneMergeDecision.action(
            [
                walker_1,
                walker_2,
                walker_3,
            ],
            [
                [
                    CloneMergeDecisionRecord(
                        **{
                            "decision_id": CloneMergeDecisionEnum.CLONE,
                            "target_idxs": [0, 2],
                        }
                    ),
                    CloneMergeDecisionRecord(
                        **{
                            "decision_id": CloneMergeDecisionEnum.KEEP_MERGE,
                            "target_idxs": [1],
                        }
                    ),
                    CloneMergeDecisionRecord(
                        **{
                            "decision_id": CloneMergeDecisionEnum.SQUASH,
                            "target_idxs": [1],
                        }
                    ),
                ]
            ],
        ) == [
            attrs.evolve(walker_1, weight=0.5),
            attrs.evolve(
                walker_2,
                weight=2.0,
            ),
            attrs.evolve(walker_1, weight=0.5),
        ]

    def test_parents(self):

        MultiCloneMergeDecision.parents(
            [
                CloneMergeDecisionRecord(
                    decision_id=1,
                    target_idxs=(idx,),
                )
                for idx in range(4)
            ],
        )
