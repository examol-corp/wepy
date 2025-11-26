from wepy.runners.runner import NoRunner
from wepy.walker import Walker, WalkerState


class TestNoRunner:

    def test_run_segment(self):

        runner = NoRunner()

        walker = Walker(
            state=WalkerState(a=1),
            weight=0.1,
        )
        assert (
            runner.run_segment(
                walker,
                10,
            )
            == walker
        )
