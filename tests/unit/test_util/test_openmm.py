# Third Party Library
import numpy as np
import openmm

# First Party Library
from wepy.util.openmm import array3d_to_vec3, vec3_to_array3d


def test_array3d_to_vec3():
    assert tuple(
        array3d_to_vec3(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
    ) == tuple(
        [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.0, 0.0, 0.0),
        ]
    )


def test_vec3_to_array3d():

    assert np.array_equal(
        vec3_to_array3d(
            tuple(
                [
                    openmm.Vec3(0.0, 0.0, 0.0),
                    openmm.Vec3(0.0, 0.0, 0.0),
                ]
            )
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
