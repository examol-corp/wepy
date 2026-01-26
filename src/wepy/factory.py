# Standard Library
from typing import Generic, Protocol, TypeVar

GeneratedType_ = TypeVar("GeneratedType_")


class Factory(Protocol, Generic[GeneratedType_]):

    @classmethod
    def type(cls) -> type[GeneratedType_]: ...

    def __call__(self) -> GeneratedType_: ...
