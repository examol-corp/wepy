import math
from typing import Literal, TypedDict
from wepy.missing import MISSING
from wepy.walker import (
    clone,
    squash,
    split,
    keep_merge,
    merge,
    Walker,
    WalkerState,
)

import attrs

# attrs provide the __eq__ method

MockKeys = Literal["a", "b"]
MockDataValue = int | str
class MockData(TypedDict):
    a: int
    b: str

@attrs.define
class MockWalkerState(WalkerState):
    a: int
    b: str

    def __getitem__(self, key: MockKeys) -> MockDataValue:
        if (value := getattr(self, key, MISSING)) is MISSING:
            raise KeyError(f"'key' '{key}' not found")
        else:
            return value

    def dict(self) -> MockData:
        return attrs.asdict(self)
        

class TestWalkerState:

    def test___init__(self):

        MockWalkerState(a=1, b="hello")

    def test___getitem__(self):

        assert MockWalkerState(a=1, b="hello")["a"] == 1
        assert MockWalkerState(a=1, b="hello")["b"] == "hello"

    def test___eq__(self):

        assert MockWalkerState(a=1, b="hello") == MockWalkerState(a=1, b="hello")
        assert MockWalkerState(a=1, b="hello") != MockWalkerState(a=100, b="hello")

    def test_dict(self):
        assert MockWalkerState(a=1, b="hello").dict() == {
            "a": 1,
            "b": "hello",
        }


class TestWalker:

    def test___init__(self):

        state = MockWalkerState(a=1, b="hello")
        walker = Walker(
            state=state,
            weight=0.1,
        )

        assert walker.weight == 0.1
        assert walker.state is state

    def test___eq__(self):
        assert Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.1,
        ) == Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.1,
        )

        assert Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.1,
        ) != Walker(
            state=MockWalkerState(a=100, b="hello"),
            weight=0.1,
        )

        assert Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.1,
        ) != Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.05,
        )

def test_clone():
    state = MockWalkerState(a=1, b="hello")
    walker = Walker(
        state=state,
        weight=0.1,
    )

    clones = clone(walker, 1)
    assert len(clones) == 2

    assert clones[0] == Walker(
        state=state,
        weight=0.05,
    )
    assert clones[1] == Walker(
        state=state,
        weight=0.05,
    )

def test_squash():

    walker_a = Walker(
        state=MockWalkerState(a=1, b="hello"),
        weight=0.1,
    )

    walker_b = Walker(
        state=MockWalkerState(a=10, b="hello"),
        weight=0.1,
    )

    assert squash(walker_a, walker_b) == Walker(
        state=MockWalkerState(a=10, b="hello"),
        weight=0.2,
    )

    assert squash(walker_b, walker_a) == Walker(
        state=MockWalkerState(a=1, b="hello"),
        weight=0.2,
    )

def test_split():

    walker = Walker(
        state=MockWalkerState(a=1, b="hello"),
        weight=0.1,
    )

    assert split(walker, 2) == [
        Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.05,
        ),
        Walker(
            state=MockWalkerState(a=1, b="hello"),
            weight=0.05,
        ),
    ]


def test_keep_merge():
    walkers = [
        Walker(
            state=MockWalkerState(a=10, b="hello"),
            weight=0.1,
        ),
        Walker(
            state=MockWalkerState(a=20, b="hello"),
            weight=0.1,
        ),
    ]

    assert keep_merge(walkers, 0) == Walker(
        state=MockWalkerState(a=10, b="hello"),
        weight=0.2,
    )

    assert keep_merge(walkers, 1) == Walker(
        state=MockWalkerState(a=20, b="hello"),
        weight=0.2,
    )


def test_merge():
    walkers = [
        Walker(
            state=MockWalkerState(a=10, b="hello"),
            weight=0.1,
        ),
        Walker(
            state=MockWalkerState(a=20, b="hello"),
            weight=0.1,
        ),
    ]

    merged_walker, keep_idx = merge(walkers)
    assert merged_walker.weight == 0.2

    assert walkers[keep_idx].state == merged_walker.state
