import pytest
from wepy.runners.mock import MockRunner, MockState, MockTask, MockError

def test_MockTask():

    assert MockTask(
        MockRunner(),
        10,
        fail=False,
    )(MockState(10)) == MockState(20)

    with pytest.raises(MockError):
        MockTask(
            MockRunner(),
            10,
            fail=True,
        )(MockState(10))
    
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
