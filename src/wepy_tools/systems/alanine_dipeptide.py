# Standard Library
import importlib.resources

# Third Party Library
import attrs
import mdtraj
import numpy as np
import openmm
import openmm.app

# First Party Library
from wepy.resampling.distances.base import Distance
from wepy.runners.openmm import OpenMMState, OpenMMStateWrapper
from wepy.util.mdtraj import json_to_mdtraj_topology, traj_fields_to_mdtraj
from wepy.walker import WalkerState


class AlanineDipeptideExplicitSystem:

    system: openmm.System
    topology: openmm.app.Topology
    mdtraj_top: mdtraj.Topology
    json_top: str
    state_wrapper: OpenMMStateWrapper
    state: OpenMMState

    def __init__(self) -> None:

        # load the system and state information for the simulation
        ala_files = importlib.resources.files(
            "wepy_tools.systems.data.alanine_dipeptide_explicit"
        )
        system_xml_path = ala_files / "alanine-dipeptide-explicit.system.omm.xml"
        state_xml_path = ala_files / "alanine-dipeptide-explicit.state.omm.xml"
        top_json_path = ala_files / "alanine-dipeptide-explicit.top.json"

        self.system = openmm.XmlSerializer.deserialize(system_xml_path.read_text())

        self.state_wrapper = OpenMMStateWrapper.from_xml(state_xml_path.read_text())
        self.state = OpenMMState.from_state_wrapper(self.state_wrapper)

        self.json_top = top_json_path.read_text()
        self.mdj_top = json_to_mdtraj_topology(self.json_top)
        self.topology = self.mdj_top.to_openmm()


@attrs.define
class AlanineDipeptideRamachandranDistanceImage(WalkerState):

    phis: np.typing.ArrayLike
    psis: np.typing.ArrayLike


@attrs.define
class AlanineDipeptideRamachandranDistance(Distance):
    # the parsed JSON topology of plain python objects
    topology: str

    def image(self, state: OpenMMState) -> AlanineDipeptideRamachandranDistanceImage:

        _unit = state["positions"].unit
        state_dict = {
            # traj shape to match interface requirements
            key: np.array([quantity.value_in_unit(_unit)])
            for key, quantity in state.to_dict().items()
            if key in {"positions", "box_vectors"}
        }
        traj = traj_fields_to_mdtraj(
            state_dict,
            self.topology,
        )

        is_periodic = "box_vectors" in state

        _, phis = mdtraj.compute_phi(
            traj,
            periodic=is_periodic,
            opt=True,
        )
        _, psis = mdtraj.compute_psi(
            traj,
            periodic=is_periodic,
            opt=True,
        )

        return AlanineDipeptideRamachandranDistanceImage(
            phis=phis,
            psis=psis,
        )

    def image_distance(
        self,
        image_a: AlanineDipeptideRamachandranDistanceImage,
        image_b: AlanineDipeptideRamachandranDistanceImage,
    ) -> float:

        angles_a = np.concatenate((image_a.phis, image_a.psis))
        angles_b = np.concatenate((image_b.phis, image_b.psis))

        # compute the circular difference
        deltas = np.atan2(
            np.sin(angles_a - angles_b),
            np.cos(angles_a - angles_b),
        )

        return np.sqrt(np.sum(deltas**2))
