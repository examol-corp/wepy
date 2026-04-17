"""OpenMM reporters used in wepy."""

# Standard Library
import abc
import logging
from typing import Literal, NotRequired, TypedDict, get_args

# Third Party Library
import openmm as omm
import openmm.app as omma

logger = logging.getLogger(__name__)


class OpenMMReporterNextReport(TypedDict):

    steps: int
    include: list[str]
    periodic: NotRequired[bool | None] = False


OpenMMGetStateKeys = Literal[
    "positions",
    "velocities",
    "forces",
    "energy",
    "parameters",
    "parameterDerivatives",
    "integratorParameters",
]
OPENMM_GET_STATE_KEYS: frozenset[OpenMMGetStateKeys] = frozenset(
    get_args(OpenMMGetStateKeys)
)


class OpenMMReporter(metaclass=abc.ABCMeta):
    """ABC for openmm.app Reporter.

    Documents the interface for openmm.app Reporter compatible classes.
    """

    def describeNextReport(
        self, simulation: omma.Simulation
    ) -> OpenMMReporterNextReport:
        raise NotImplementedError

    def report(
        self,
        simulation: omma.Simulation,
        state: omm.State,
    ) -> None:

        raise NotImplementedError
