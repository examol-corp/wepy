import attrs
from wepy.util.attrs import AttrsMappingMixin

@attrs.define
class Thing(AttrsMappingMixin):
    a: int
    b: str

class Test_AttrsMappingMixin:

    def test___init__(self):

        Thing(a=1, b="hello")

    def test___len__(self):

        assert len(Thing(a=1, b="hello")) == 2

    def test___getitem__(self):

        assert Thing(a=1, b="hello")["a"] == 1
        assert Thing(a=1, b="hello")["b"] == "hello"

    def test___eq__(self):

        assert Thing(a=1, b="hello") == Thing(a=1, b="hello")

    def test__ne__(self):
        assert Thing(a=1, b="hello") != Thing(a=100, b="hello")
        assert Thing(a=1, b="hello") != Thing(a=1, b="goodbye")

    def test___contains__(self):

        t = Thing(a=1, b="hello")
        assert "a" in t
        assert "b" in t

    def test___iter__(self):
        t = Thing(a=1, b="hello")
        t_it = iter(t)
        _vs = set()
        _vs.add(next(t_it))
        _vs.add(next(t_it))
        assert _vs == {"a", "b"}


    def test_keys(self):
        t = Thing(a=1, b="hello")
        assert set(t.keys()) == {"a", "b"}


    def test_values(self):
        t = Thing(a=1, b="hello")
        assert set(t.values()) == {1, "hello"}


    def test_items(self):
        t = Thing(a=1, b="hello")
        assert set(t.items()) == {
            ("a", 1),
            ("b", "hello"),
        }

    def test_get(self):
        t = Thing(a=1, b="hello")
        assert t.get("a") == 1
        assert t.get("b") == "hello"
