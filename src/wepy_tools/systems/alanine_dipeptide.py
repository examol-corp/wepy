from importlib.resources import files

import openmm
import openmm.app
import openmm.unit

DEFAULT_EWALD_ERROR_TOLERANCE = 1.0e-5
DEFAULT_CUTOFF_DISTANCE = 10.0 * openmm.unit.angstroms
DEFAULT_SWITCH_WIDTH = 1.5 * openmm.unit.angstroms

class AlanineDipeptideExplicit:
    """Alanine dipeptide ff96 in TIP3P explicit solvent.

    Parameters
    ----------
    constraints : optional, default=openmm.app.HBonds
    rigid_water : bool, optional, default=True
    nonbondedCutoff : Quantity, optional, default=9.0 * unit.angstroms
    use_dispersion_correction : bool, optional, default=True
        If True, the long-range disperson correction will be used.
    nonbondedMethod : openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.
    cutoff : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_CUTOFF_DISTANCE
        Cutoff distance
    switch_width : openmm.unit.Quantity with units compatible with angstroms, optional, default = DEFAULT_SWITCH_WIDTH
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    ewaldErrorTolerance : float, optional, default=DEFAULT_EWALD_ERROR_TOLERANCE
           The Ewald or PME tolerance.

    Examples
    --------

    >>> alanine = AlanineDipeptideExplicit()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(
            self,
            constraints=openmm.app.HBonds,
            rigid_water=True,
            nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE,
            use_dispersion_correction=True,
            nonbondedMethod=openmm.app.PME,
            hydrogenMass=None,
            switch_width=DEFAULT_SWITCH_WIDTH,
            ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE,
    ):

        prmtop_filename = files("wepy_tools.systems.data.alanine_dipeptide_explicit") / "alanine-dipeptide.prmtop"
        crd_filename = files("wepy_tools.systems.data.alanine_dipeptide_explicit") / "alanine-dipeptide.crd"

        # Initialize system.
        prmtop = openmm.app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(
            constraints=constraints,
            nonbondedMethod=nonbondedMethod,
            rigidWater=rigid_water,
            nonbondedCutoff=nonbondedCutoff,
            hydrogenMass=hydrogenMass,
        )

        # Extract topology
        self.topology = prmtop.topology

        # Set dispersion correction use.
        forces = {
            system.getForce(index).__class__.__name__: system.getForce(index)
            for index
            in range(system.getNumForces())
        }
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

        # Read positions.
        inpcrd = openmm.app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        self.system, self.positions = system, positions
