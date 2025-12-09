from typing import Any
from importlib.resources import files

import attrs
import numpy as np
import openmm
import openmm.app
import openmm.unit
import mdtraj

from wepy.walker import WalkerState
from wepy.runners.openmm import OpenMMState
from wepy.resampling.distances.distance import Distance
from wepy.util.mdtraj import traj_fields_to_mdtraj

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
            key : np.array([quantity.value_in_unit(_unit)])
            for key, quantity
            in state.to_dict().items()
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
