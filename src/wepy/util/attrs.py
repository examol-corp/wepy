from typing import Any, Iterator
from collections.abc import Mapping

import attrs

from wepy.missing import MISSING

class AttrsMappingMixin(Mapping[str, object]):
    """A convenient mixin for implementing the WalkerState interface
    for attrs classes."""

    def __len__(self) -> int:

        return len(attrs.fields(type(self)))

    def __getitem__(self, key: str) -> Any:
        if (value := getattr(self, key, MISSING)) is MISSING:
            raise KeyError(f"'key' '{key}' not found")
        else:
            return value

    def __iter__(self) -> Iterator[str]:

        for field in attrs.fields(type(self)):
            yield field.name
