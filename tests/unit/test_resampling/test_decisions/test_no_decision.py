from wepy.walker import Walker, WalkerState
from wepy.runners.mock import MockState
from wepy.resampling.decisions.no_decision import NoDecision, NothingDecisionEnum, NoDecisionRecord


class TestNoDecision:

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
                        NoDecisionRecord(**{
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idx": 0,
                        }),
                        NoDecisionRecord(**{
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idx": 1,
                        }),
                    ]
                ],
            )
            == walkers
        )

        NoDecision.action(
            walkers,
            [
                [
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 1,
                    }),
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 0,
                    }),
                ]
            ],
        ) == [walker_2, walker_1]

    def test_parents(self):

        assert NoDecision.parents(
                [
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 0,
                    }),
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 1,
                    }),
                ]
        ) == [0, 1]

        assert NoDecision.parents(
                [
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 1,
                    }),
                    NoDecisionRecord(**{
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idx": 0,
                    }),
                ]
        ) == [1, 0]
