import json

import numpy as np
import mdtraj

from wepy.util.mdtraj import mdtraj_to_json_topology
from wepy.runners.openmm import OpenMMState
from wepy_tools.systems.alanine_dipeptide import AlanineDipeptideExplicitSystem, AlanineDipeptideRamachandranDistance


def test_AlanineDipeptideExplicitSystem():

    AlanineDipeptideExplicitSystem()

class Test_AlanineDipeptideRamachandranDistance:

    def test_image(self):

        ala_sys = AlanineDipeptideExplicitSystem()

        distance = AlanineDipeptideRamachandranDistance(topology=ala_sys.json_top)

        image = distance.image(ala_sys.state)

    def test_image_distance(self):

        ala_sys = AlanineDipeptideExplicitSystem()

        distance = AlanineDipeptideRamachandranDistance(topology=ala_sys.json_top)

        image_a = distance.image(ala_sys.state)

        assert np.isclose(distance.image_distance(image_a, image_a), 0.)

        # then make a jittered atom positions to get something a little
        # different to compare
        jitter_positions = ala_sys.state.positions + np.random.uniform(
            -0.01,
            0.01,
            size=ala_sys.state.positions.shape,
        ) * ala_sys.state.positions.unit

        jitter_state = OpenMMState.from_dwim(
            positions=jitter_positions,
            box_vectors=ala_sys.state.box_vectors,
        )

        jitter_image = distance.image(jitter_state)

        assert not np.isclose(
            distance.image_distance(
                image_a,
                jitter_image,
            ),
            0.,
        )

        # test it is symmetric
        assert np.isclose(
            distance.image_distance(
                image_a,
                jitter_image,
            ),
            distance.image_distance(
                jitter_image,
                image_a,
            ),
        )
