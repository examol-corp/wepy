from wepy.runners.mock import MockRunner, MockState

class TestMockRunner:

    def test_pre_cycle(self):
        MockRunner().pre_cycle()

    def test_post_cycle(self):
        MockRunner().post_cycle()


    def test_run_segment(self):

        assert MockRunner().run_segment(
            MockState(0),
            10,
            0,
        ) == MockState(10)
