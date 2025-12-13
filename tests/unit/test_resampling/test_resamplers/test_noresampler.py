# First Party Library
from wepy.resampling.decisions.no_decision import NothingDecisionEnum
from wepy.resampling.resamplers.noresampler import NoResampler, NoResamplerFactory, NoResamplerResamplingRecord, NoResamplerResamplerRecord
from wepy.runners.mock import MockState
from wepy.walker import Walker

def test_NoResamplerResamplingRecord():

    assert NoResamplerResamplingRecord(0, (1,)) == NoResamplerResamplingRecord(0, (1,))

class Test_NoResampler:

    def test_resample(self):

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

        walkers = [walker_1, walker_2, walker_3]

        resampler = NoResampler()

        assert resampler.resample(walkers) == (
            walkers,
            [
                    NoResamplerResamplingRecord(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[0],
                    ),
                    NoResamplerResamplingRecord(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[1],
                    ),
                    NoResamplerResamplingRecord(
                        decision_id=NothingDecisionEnum.NOTHING.value,
                        target_idxs=[2],
                    ),
            ],
            [NoResamplerResamplerRecord()],
        )


def test_NoResamplerFactory():

    factory = NoResamplerFactory()
    factory(num_cores=1)
