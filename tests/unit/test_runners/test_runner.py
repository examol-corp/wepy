import attrs
from wepy.runners.runner import NoRunner
from wepy.walker import Walker, WalkerState


class TestNoRunner:

    def test_run_segment(self):

        # concrete state to use
        @attrs.define
        class SomeState(WalkerState):
            a: int

            def __getitem__(self, key: str) -> int:

                if key != "a":
                    raise KeyError(f"Invalid key '{key}'")

                return self.a

            def dict(self) -> dict[str, int]:
                return {"a" : self.a}

        runner = NoRunner()

        walker = Walker(
            state=SomeState(a=1),
            weight=0.1,
        )
        assert (
            runner.run_segment(
                walker,
                10,
            )
            == walker
        )
