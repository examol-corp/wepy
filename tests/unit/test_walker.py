
import math
from wepy.walker import (
    split,
    keep_merge,
    merge,
    Walker,
    WalkerState,
)

class TestWalkerState:

    def test___init__(self):

        WalkerState(a=1, b="hello")

    def test___getitem__(self):

        assert WalkerState(a=1, b="hello")['a'] == 1
        assert WalkerState(a=1, b="hello")['b'] == "hello"

    def test___eq__(self):

        assert WalkerState(a=1, b="hello") == WalkerState(a=1, b="hello")
        assert WalkerState(a=1, b="hello") != WalkerState(a=100, b="hello")
        
    def test_dict(self):
        assert WalkerState(a=1, b="hello").dict() == {
            "a" : 1,
            "b" : "hello",
        }

class TestWalker:

    def test___init__(self):

        state = WalkerState(a=1, b="hello")
        walker = Walker(
            state=state,
            weight=0.1,
        )

        assert walker.weight == 0.1
        assert walker.state is state

    def test___eq__(self):
        assert Walker(
            state=WalkerState(a=1, b="hello"),
            weight=0.1,
        ) == Walker(
            state=WalkerState(a=1, b="hello"),
            weight=0.1,
        )

        assert Walker(
            state=WalkerState(a=1, b="hello"),
            weight=0.1,
        ) != Walker(
            state=WalkerState(a=100, b="hello"),
            weight=0.1,
        )

        assert Walker(
            state=WalkerState(a=1, b="hello"),
            weight=0.1,
        ) != Walker(
            state=WalkerState(a=1, b="hello"),
            weight=0.05,
        )
        
    def test_clone(self):
        state = WalkerState(a=1, b="hello")
        walker = Walker(
            state=state,
            weight=0.1,
        )

        clones = walker.clone(1)
        assert len(clones) == 2

        assert clones[0] == Walker(
            state=state,
            weight=0.05,
        )
        assert clones[1] == Walker(
            state=state,
            weight=0.05,
        )



    def test_squash(self):

        walker_a = Walker(
            state=WalkerState(a=1),
            weight=0.1,
        )

        walker_b = Walker(
            state=WalkerState(a=10),
            weight=0.1,
        )

        assert walker_a.squash(walker_b) == Walker(
            state=WalkerState(a=10),
            weight=0.2,
        )

        assert walker_b.squash(walker_a) == Walker(
            state=WalkerState(a=1),
            weight=0.2,
        )

        
        

    def test_merge(self):
        walker_a = Walker(
            state=WalkerState(a=1),
            weight=0.1,
        )

        other_walkers = [
            Walker(
                state=WalkerState(a=10),
                weight=0.1,
            ),
            Walker(
                state=WalkerState(a=20),
                weight=0.1,
            ),
        ]

        assert math.isclose(walker_a.merge(other_walkers).weight, 0.3)

        

def test_split():

    walker = Walker(
        state=WalkerState(a=1),
        weight=0.1,
    )

    assert split(walker, 2) == [
        Walker(
            state=WalkerState(a=1),
            weight=0.05,
        ),
        Walker(
            state=WalkerState(a=1),
            weight=0.05,
        )
    ]
    

def test_keep_merge():
    walkers = [
        Walker(
            state=WalkerState(a=10),
            weight=0.1,
        ),
        Walker(
            state=WalkerState(a=20),
            weight=0.1,
        ),
    ]

    assert keep_merge(walkers, 0) == Walker(
            state=WalkerState(a=10),
            weight=0.2,
        )

    assert keep_merge(walkers, 1) == Walker(
            state=WalkerState(a=20),
            weight=0.2,
        )
    
def test_merge():
    walkers = [
        Walker(
            state=WalkerState(a=10),
            weight=0.1,
        ),
        Walker(
            state=WalkerState(a=20),
            weight=0.1,
        ),
    ]

    merged_walker, keep_idx = merge(walkers)
    assert merged_walker.weight == 0.2

    assert walkers[keep_idx].state == merged_walker.state
