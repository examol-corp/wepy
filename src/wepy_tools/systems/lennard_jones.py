# Third Party Library
import attrs
import numpy as np
import openmm
import openmm.app
import openmm.unit
from scipy.spatial.distance import euclidean

# First Party Library
from wepy.resampling.distances.base import Distance
from wepy.runners.openmm import OpenMMState


class LennardJonesPair:
    """Create a pair of Lennard-Jones particles.

    Parameters
    ----------
    mass : simtk.unit.Quantity with units compatible with amu, optional, default=39.9*amu
       The mass of each particle.
    epsilon : simtk.unit.Quantity with units compatible with kilojoules_per_mole, optional, default=1.0*kilocalories_per_mole
       The effective Lennard-Jones sigma parameter.
    sigma : simtk.unit.Quantity with units compatible with nanometers, optional, default=3.350*angstroms
       The effective Lennard-Jones sigma parameter.

    Examples
    --------
    Create Lennard-Jones pair.

    >>> test = LennardJonesPair()
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    Create Lennard-Jones pair with different well depth.

    >>> test = LennardJonesPair(epsilon=11.0*unit.kilocalories_per_mole)
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    Create Lennard-Jones pair with different well depth and sigma.

    >>> test = LennardJonesPair(epsilon=7.0*unit.kilocalories_per_mole, sigma=4.5*unit.angstroms)
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    """

    def __init__(
        self,
        mass=39.9 * openmm.unit.amu,
        sigma=3.350 * openmm.unit.angstrom,
        epsilon=10.0 * openmm.unit.kilocalories_per_mole,
    ):

        # Store parameters
        self.mass = mass
        self.sigma = sigma
        self.epsilon = epsilon

        # Charge must be zero.
        charge = 0.0 * openmm.unit.elementary_charge

        # Create an empty system object.
        system = openmm.System()

        # Create a NonbondedForce object with no cutoff.
        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        # Create positions.
        positions = openmm.unit.Quantity(
            np.zeros([2, 3], np.float32), openmm.unit.angstrom
        )
        # Move the second particle along the x axis to be at the potential minimum.
        positions[1, 0] = 2.0 ** (1.0 / 6.0) * sigma

        # Create first particle.
        system.addParticle(mass)
        force.addParticle(charge, sigma, epsilon)

        # Create second particle.
        system.addParticle(mass)
        force.addParticle(charge, sigma, epsilon)

        # Add the nonbonded force.
        system.addForce(force)

        # Store system and positions.
        self.system, self.positions = system, positions

        # Store ligand and receptor particle indices.
        self.ligand_indices = [0]
        self.receptor_indices = [1]

        # Create topology.
        topology = openmm.app.Topology()
        element = openmm.app.Element.getBySymbol("Ar")
        chain = topology.addChain()
        residue = topology.addResidue("Ar", chain)
        topology.addAtom("Ar", element, residue)
        residue = topology.addResidue("Ar", chain)
        topology.addAtom("Ar", element, residue)
        self.topology = topology


@attrs.define
class PairDistanceImage:
    positions: np.typing.ArrayLike


class PairDistance(Distance):
    def __init__(self, metric=euclidean):
        self.metric = metric

    def image(self, state: OpenMMState) -> PairDistanceImage:
        return state.positions

    def image_distance(
        self, image_a: PairDistanceImage, image_b: PairDistanceImage
    ) -> float:
        dist_a = self.metric(image_a[0], image_a[1])
        dist_b = self.metric(image_b[0], image_b[1])

        return np.abs(dist_a - dist_b)
