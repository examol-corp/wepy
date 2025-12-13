import pytest
# First Party Library
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NoDecisionRecord,
    NothingDecisionEnum,
)
from wepy.runners.mock import MockState
from wepy.walker import Walker

class Test_NoDecisionRecord:

    def test___init__(self):

        NoDecisionRecord(
            decision_id=0,
            target_idx=1,
        )

        with pytest.raises(ValueError):
            NoDecisionRecord(
                decision_id=1,
                target_idx=1,
            )

        with pytest.raises(ValueError):
            NoDecisionRecord(
                decision_id=0,
                target_idx=-1,
            )

    def test_to_dict(self):

        assert NoDecisionRecord(
            decision_id=0,
            target_idx=0
        ).to_dict() == {
            "decision_id" : 0,
            "target_idx" : 0,
        }

class Test_NoDecision:

    def test_action(self):

        walker_1 = Walker(
            state=MockState(a=1),
            weight=1.0,
        )

        walker_2 = Walker(
            state=MockState(a=2),
            weight=1.0,
        )

        walkers = [
            walker_1,
            walker_2,
        ]

        assert (
            NoDecision.action(
                walkers,
                [
                    [
                        NoDecisionRecord(
                            **{
                                "decision_id": NothingDecisionEnum.NOTHING,
                                "target_idx": 0,
                            }
                        ),
                        NoDecisionRecord(
                            **{
                                "decision_id": NothingDecisionEnum.NOTHING,
                                "target_idx": 1,
                            }
                        ),
                    ]
                ],
            )
            == walkers
        )

        NoDecision.action(
            walkers,
            [
                [
                    NoDecisionRecord(
                        **{
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idx": 1,
                        }
                    ),
                    NoDecisionRecord(
                        **{
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idx": 0,
                        }
                    ),
                ]
            ],
        ) == [walker_2, walker_1]

    def test_parents(self):

        assert NoDecision.parents(
            [
                NoDecisionRecord(
                    **{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 0,
                    }
                ),
                NoDecisionRecord(
                    **{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 1,
                    }
                ),
            ]
        ) == [0, 1]

        assert NoDecision.parents(
            [
                NoDecisionRecord(
                    **{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 1,
                    }
                ),
                NoDecisionRecord(
                    **{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 0,
                    }
                ),
            ]
        ) == [1, 0]
