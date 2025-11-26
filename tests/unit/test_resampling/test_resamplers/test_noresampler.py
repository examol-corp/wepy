from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.resampling.decisions.no_decision import NothingDecisionEnum
from wepy.walker import Walker, WalkerState


class TestNoResampler:

    def test_resample(self):

        walker_1 = Walker(
            state=WalkerState(a=1),
            weight=1.0,
        )

        walker_2 = Walker(
            state=WalkerState(a=2),
            weight=1.0,
        )
        walker_3 = Walker(
            state=WalkerState(a=3),
            weight=1.0,
        )

        walkers = [walker_1, walker_2, walker_3]

        resampler = NoResampler()

        assert resampler.resample(walkers) == (
            walkers,
            [
                [
                    dict(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[0],
                    ),
                    dict(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[1],
                    ),
                    dict(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[2],
                    ),
                ]
            ],
            [{}],
        )
