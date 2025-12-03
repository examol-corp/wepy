from typing import Generator
from collections.abc import Iterable
import numpy as np
import openmm

def array3d_to_vec3(array: np.typing.ArrayLike) -> Generator[openmm.Vec3, None, None]:

    for row in array:
        yield openmm.Vec3(*row.tolist())

def vec3_to_array3d(vec3s: Iterable[openmm.Vec3]) -> np.typing.ArrayLike:

    vs = []
    for vec3 in vec3s:
        vs.append(
            (
                vec3.x,
                vec3.y,
                vec3.z,
            )
        )

    return np.array(vs)
        
