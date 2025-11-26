from wepy.walker import Walker, WalkerState
from wepy.resampling.decisions.no_decision import NoDecision, NothingDecisionEnum


class TestNoDecision:

    def test_action(self):

        walker_1 = Walker(
            state=WalkerState(a=1),
            weight=1.0,
        )

        walker_2 = Walker(
            state=WalkerState(a=2),
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
                        {
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idxs": [0],
                        },
                        {
                            "decision_id": NothingDecisionEnum.NOTHING,
                            "target_idxs": [1],
                        },
                    ]
                ],
            )
            == walkers
        )

        NoDecision.action(
            walkers,
            [
                [
                    {
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idxs": [1],
                    },
                    {
                        "decision_id": NothingDecisionEnum.NOTHING,
                        "target_idxs": [0],
                    },
                ]
            ],
        ) == [walker_2, walker_1]
