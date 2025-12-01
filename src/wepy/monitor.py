"""Interface definition for simulation monitors."""

from wepy.walker import Walker


class Monitor:

    def init(self) -> None: ...

    def cycle_monitor(self, walkers: list[Walker]) -> None: ...

    def cleanup(self) -> None: ...
