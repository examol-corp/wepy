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
        


def triclinic_volume_vec3_quantity(box_vectors: list[openmm.unit.Quantity]) -> openmm.unit.Quantity:

    return np.dot(box_vectors[0], np.cross(box_vectors[1], box_vectors[2]))

def format_box_vectors_line(box_vectors: list[openmm.unit.Quantity]) -> str:
    unit = box_vectors[0].unit

    vec_strs = []
    for vec in box_vectors:
        mags = [
            q.value_in_unit(q.unit)
            for q in vec
        ]
        vec_s = f"{mags[0]:.3f}, {mags[1]:.3f}, {mags[2]:.3f}"
        vec_strs.append(vec_s)

    s = f"({vec_strs[0]}) ({vec_strs[1]}) ({vec_strs[2]}) {unit}"

    return s
