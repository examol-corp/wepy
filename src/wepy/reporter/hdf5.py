# Standard Library
from pathlib import Path
import logging
from typing import Self, TypedDict, Literal, Generic, TypeVar, Any

# Standard Library

# Third Party Library
import numpy as np
import openmm.unit

# First Party Library
from wepy.hdf5 import WepyHDF5
from wepy.reporter.base import (
    SimComponentArgs,
    CycleReportDict,
)
from wepy.storage.protocol import (
    RecordFieldShapeSpec,
    RecordFieldDtype,
)
from wepy.reporter.file import FileReporterABC, FileMode
from wepy.util.json_top import json_top_atom_count
from wepy.walker import Walker, WalkerState, WalkerStateBox
from wepy.resampling.resamplers.resampler import Resampler
from wepy.boundary_conditions.boundary import BoundaryConditions
from wepy.typing import Shape, Idxs, IdxArray
from wepy.storage.protocol import Record

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)
ResamplingRecord_ = TypeVar("ResamplingRecord_", bound=Record)
ResamplerRecord_ = TypeVar("ResamplerRecord_", bound=Record)

WarpingRecord_ = TypeVar("WarpingRecord_", bound=Record)
BCRecord_ = TypeVar("BCRecord_", bound=Record)
ProgressRecord_ = TypeVar("ProgressRecord_", bound=Record)

class UnitError(Exception):
    pass

# TODO: support for pint
Quantity = openmm.unit.Quantity

class WepyHDF5Reporter(
        FileReporterABC,
        Generic[
            WalkerState_,
            ResamplingRecord_,
            ResamplerRecord_,
            WarpingRecord_,
            BCRecord_,
            ProgressRecord_,
        ],
):
    """Reporter for generating an HDF5 format (WepyHDF5) data file from
    simulations.

    This is the most important reporter as it is the principle output
    format for storing weighted ensemble simulation data.

    Files generated with this reporter can be opened using the
    wepy.hdf5.WepyHDF5 class.

    See Also
    --------
    wepy.hdf5.WepyHDF5


    """

    # this is the name of the dataset that the all atoms will be saved
    # under in the HDF5 alt_reps group
    ALL_ATOMS_REP_KEY = "all_atoms"

    FILE_ORDER = ("wepy_hdf5_path",)

    # this is the suggested extension for naming WepyHDF5 files used
    # by this reporter, e.g. results.wepy.h5
    SUGGESTED_EXTENSIONS = ("wepy.h5",)

    # static attributes
    swmr_mode: bool
    save_fields: tuple[str, ...] | None
    _sparse_fields: dict[str, int]
    _feature_shapes: dict[str, RecordFieldShapeSpec] | None
    _feature_dtypes: dict[str, RecordFieldDtype] | None
    _n_dims: int
    resampling_fields: tuple[str, ...]
    decision_enum_dict: dict[str, int]
    resampler_fields: tuple[str, ...] | None
    warping_fields: tuple[str, ...] | None
    progress_fields: tuple[str, ...] | None
    bc_fields: tuple[str, ...] | None
    resampling_records: tuple[str, ...] | None
    resampler_records: tuple[str, ...] | None
    bc_records: tuple[str, ...] | None
    warping_records: tuple[str, ...] | None
    progress_records: tuple[str, ...] | None
    main_rep_idxs: IdxArray | None
    alt_reps_to_save: list[str]
    alt_reps_idxs: dict[str, IdxArray]
    _n_atoms: int
    _all_atom_idxs: IdxArray
    _sparse_fields: dict[str, int]
    units: dict[str, openmm.unit.Unit]

    # stateful attributes
    wepy_h5: WepyHDF5 | None
    wepy_run_idx: int | None
    
    _tmp_topology: str | None

    def __init__(
        self,
        file_path: Path,
        topology: str,
        # Resampling features
        decision_enum_dict: dict[str, int],
        resampling_fields: tuple[str, ...],
        swmr_mode: bool = False,
        save_fields: tuple[str, ...] | None = None,
        units: dict[str, openmm.unit.Unit] | None = None,
        sparse_fields: dict[str, int | Literal[Ellipsis]] | None = None,
        n_dims: int = 3,
        main_rep_idxs: Idxs | None = None,
        all_atoms_rep_freq: int | None = None,
        alt_reps: dict[str, tuple[Idxs, int]] | None = None,
        # TOREV: are these feature fields actually needed for the main
        # trajectories? I never used them. If they are useful they
        # should be derived from runner metadata in the common case. I
        # think in most cases they are determined dynamically, so this
        # needs to be amended.
        feature_shapes: dict[str, RecordFieldShapeSpec] | None = None,
        feature_dtypes: dict[str, RecordFieldDtype] | None = None,
        # Resampling optionals
        resampling_records: tuple[str, ...] | None = None,
        # Resampler fields are optional
        resampler_fields: tuple[str, ...] | None = None,
        resampler_records: tuple[str, ...] | None = None,
        # BC features, optional
        warping_fields: tuple[str, ...] | None = None,
        progress_fields: tuple[str, ...] | None = None,
        bc_fields: tuple[str, ...] | None = None,
        warping_records: tuple[str, ...] | None = None,
        bc_records: tuple[str, ...] | None = None,
        progress_records: tuple[str, ...] | None = None,
    ):
        """Constructor for the WepyHDF5Reporter.

        Parameters
        ----------
        
        save_fields : A selection of fields from the walker states to
           be stored. Allows for the ignoring of some states. If None
           all fields from states will attempted to be saved. To not
           save anything provide an empty tuple ().

        topology : str
            JSON string representing topology of system being simulated.

        units : Mapping of trajectory field names to Unit objects.

        sparse_fields : dict of str: int, optional
            List of trajectory fields that should be initialized as sparse.

        feature_shapes : Mapping of trajectory fields to their shape
            spec for initialization. Note that these are extras and the
            defaults for OpenMM MD will automatically be configured.

        feature_dtypes : Mapping of trajectory fields to their shape
            spec for initialization. Note that these are extras and the
            defaults for OpenMM MD will automatically be configured.

        n_dims : int, default: 3
            Set the number of spatial dimensions for the default
            positions trajectory field.

        alt_reps : dict of str: tuple of (list of int, int), optional
            Specifies that there will be 'alt_reps' of positions each
            named by the keys of this mapping and containing the
            indices in each value list as the first value of the tuple
            and the second value being the frequency at which this
            field gets saved. Setting `all_atoms_rep_freq` is the
            equivalent of setting an entry {'all_atoms' : ([...],
            `all_atoms_rep_freq`)}.

        main_rep_idxs : list of int, optional
            The indices of atom positions to save as the main 'positions'
            trajectory field. Defaults to all atoms.

        all_atoms_rep_freq : int, optional
            The frequency at which to set an 'alt_rep' for all of the
            atoms in a simulation. Will be set as the field
            'alt_rep/all_atoms'.

        swmr_mode : bool
           Whether to write to open the HDF5 in single-writer
           multi-reader (SWMR) mode.


        Other Parameters
        ----------------
        resampling_fields : list of str
            The names of the fields for resampling records

        decision_enum_dict : dict of str : int
            Mapping of the names of resampling decision enum to their
            integer values.

        resampler_fields : list of str
            The names of the fields for the resampler records

        warping_fields : list of str
            The names of the fields for the warping records

        progress_fields : list of str
            The names of the fields for the progress records

        bc_fields : list of str
            The names of the fields for the bounadry condition records.

        resampling_records : list of str, optional
            Names of the resampling_fields that will be used in
            table-like views.

        resampler_records : list of str, optional
            Names of the resampler_fields that will be used in
            table-like views.

        warping_records : list of str, optional
            Names of the warping_fields that will be used in
            table-like views.

        bc_records : list of str, optional
            Names of the bc_fields that will be used in table-like
            views.

        progress_records : list of str, optional
            Names of the progress_fields that will be used in
            table-like views.

        """

        # initialize inherited attributes
        super().__init__(
            file_paths=[file_path],
            # hardcode creation mode
            modes=["x"],
        )

        # set the preference for swmr mode, True or False, if this is
        # True then SWMR mode will be turned on when the file is
        # written to during reporting
        self.swmr_mode = swmr_mode

        self.wepy_run_idx = None
        self._tmp_topology = topology

        # which fields from the walker to save, if None then save all of them
        self.save_fields = save_fields

        # check sparse fields
        if sparse_fields is not None:

            if self.save_fields is None:
                raise ValueError(
                    f"The sparse fields were requested ({set(sparse_fields.keys())}) but no save fields requested."
                )
                
            _missing_save_fields = set(
                sparse_key
                for sparse_key
                in sparse_fields.keys()
                if sparse_key not in self.save_fields
            )

            if len(_missing_save_fields) > 0:
                raise ValueError(
                    f"The sparse fields were requested ({_missing_save_fields}) but are not in"
                    f" requested save fields ({self.save_fields})"
                )

            self._sparse_fields = sparse_fields

        else:
            self._sparse_fields = {}

        self._feature_shapes = feature_shapes
        self._feature_dtypes = feature_dtypes
        self._n_dims = n_dims


        # required resampling fields
        self.resampling_fields = resampling_fields
        self.decision_enum_dict = decision_enum_dict

        # optional resampler fields
        self.resampler_fields = resampler_fields
        self.resampling_records = resampling_records
        self.resampler_records = resampler_records

        # BC fields, optional
        self.warping_fields = warping_fields
        self.progress_fields = progress_fields
        self.bc_fields = bc_fields

        # the fields which are records for table like reports
        self.bc_records = bc_records
        self.warping_records = warping_records
        self.progress_records = progress_records

        # the atom indices of the whole system that will be saved as
        # the main positions representation
        self.main_rep_idxs = np.array(main_rep_idxs) if main_rep_idxs is not None else None

        # the idxs for alternate representations of the system
        # positions

        # this is a record of which alt_reps to actually save in the simulation
        self.alt_reps_to_save = []
        if alt_reps is not None:
            self.alt_reps_idxs = {
                key: np.array(idxs)
                for key, (idxs, frequence)
                in alt_reps.items()
            }

            # add the frequencies for these alt_reps to the
            # sparse_fields frequency dictionary
            for key, (idxs, freq) in alt_reps.items():

                if len(idxs) == 0:
                    raise ValueError(
                        f"No indices given for sparse field: {key}"
                    )


                alt_rep_key = "alt_reps/{}".format(key)

                # if the frequency is Ellipsis or 1 then we save it
                # every frame and don't make it sparse because that is
                # very innefficient in comparison
                if freq is Ellipsis or freq == 1 or freq == 0:
                    pass
                elif freq > 0:
                    self._sparse_fields[alt_rep_key] = freq

                else:
                    raise ValueError(
                        f"Invalid frequency specifier ({freq}) for sparse field '{key}'"
                    )

                self.alt_reps_to_save.append(key)
                self.alt_reps_idxs[key] = np.array(idxs)

        else:
            self.alt_reps_idxs = {}

        # check for alt_reps of this name because this is reserved for
        # the all_atoms flag.
        if self.ALL_ATOMS_REP_KEY in self.alt_reps_idxs:
            raise ValueError("Cannot name an alt_rep 'all_atoms'")

        # Handle the all_atoms fields. There is always a record of
        # what the all_atoms are even if there is no extra field for
        # this given in the trajectory data.

        # count the number of atoms in the topology and set the
        # alt_reps to have the full slice for all atoms
        self._n_atoms = json_top_atom_count(self._tmp_topology)
        self._all_atom_idxs = np.arange(self._n_atoms)
        self.alt_reps_idxs[self.ALL_ATOMS_REP_KEY] = self._all_atom_idxs

        # if there is a frequency for all atoms rep then we make an
        # alt_rep for the all_atoms system with the specified
        # frequency
        if all_atoms_rep_freq is not None:

            # add the frequency for this sparse fields to the
            # sparse fields dictionary
            self._sparse_fields["alt_reps/{}".format(self.ALL_ATOMS_REP_KEY)] = (
                all_atoms_rep_freq
            )

            self.alt_reps_to_save.append(self.ALL_ATOMS_REP_KEY)

        # if there are no sparse fields set it as an empty dictionary
        if self._sparse_fields is None:
            self._sparse_fields = {}

        # if units were given add them otherwise set as an empty dictionary
        if units is None:
            self.units = {}
        else:
            self.units = units

    @classmethod
    def from_components(
        self,
        file_path: Path,
        topology: str,
        resampler_class: type[Resampler],
        feature_shapes: dict[str, RecordFieldShapeSpec] | None = None,
        feature_dtypes: dict[str, RecordFieldDtype] | None = None,
        boundary_conditions_class: type[BoundaryConditions] | None = None,
        swmr_mode: bool = False,
        save_fields: tuple[str, ...] | None = None,
        units: dict[str, openmm.unit.Unit] | None = None,
        sparse_fields: dict[str, int] | None = None,
        n_dims: int = 3,
        main_rep_idxs: Idxs | None = None,
        all_atoms_rep_freq: int | None = None,
        alt_reps: dict[str, tuple[Idxs, int]]=None,
    ) -> Self:
        """Construct reporter from simulation components.

        Does introspection on components to get information. Does not
        save these objects as state.

        Parameters
        ----------

        resampler : Resampler object, optional but recommended
            The resampler being used for the simulation. Is used as a
            convenient container for a variety of constants needed for
            specifying data for the resampling records. If this is not
            given then these of the Other Parameters below must be
            specified manually: resampling_fields, decision_enum_dict,
            resampler_fields, resampling_records, resampler_records.

        boundary_conditions : BoundaryConditions object, optional but recommended
            The boundary conditions being used for the simulation. Is
            used as a convenient container for a variety of constants
            needed for specifying data for the warping and progress
            records. If this is not given then these of the Other
            Parameters below must be specified manually:
            warping_fields, progress_fields, bc_fields,
            warping_records, bc_records, progress_records

        """

        if boundary_conditions_class is not None:
            warping_fields = boundary_conditions_class.warping_fields()
            progress_fields = boundary_conditions_class.progress_fields()
            bc_fields = boundary_conditions_class.bc_fields()
            bc_records = boundary_conditions_class.bc_record_field_names()
            warping_records = boundary_conditions_class.warping_record_field_names()
            progress_records = boundary_conditions_class.progress_record_field_names()

        else:
            warping_fields = None
            progress_fields = None
            bc_fields = None
            bc_records = None
            warping_records = None
            progress_records = None
        

        return WepyHDF5Reporter(
            file_path=file_path,
            topology=topology,
            feature_shapes=feature_shapes,
            feature_dtypes=feature_dtypes,
            swmr_mode=swmr_mode,
            save_fields=save_fields,
            units=units,
            sparse_fields=sparse_fields,
            n_dims=n_dims,
            main_rep_idxs=main_rep_idxs,
            all_atoms_rep_freq=all_atoms_rep_freq,
            alt_reps=alt_reps,
            # components
            resampling_fields=resampler_class.resampling_fields(),
            decision_enum_dict=resampler_class.DECISION.enum_dict_by_name(),
            resampler_fields = resampler_class.resampler_fields(),
            resampling_records = resampler_class.resampling_record_field_names(),
            resampler_records = resampler_class.resampler_record_field_names(),
            warping_fields=warping_fields,
            progress_fields=progress_fields,
            bc_fields=bc_fields,
            bc_records=bc_records,
            warping_records=warping_records,
            progress_records=progress_records,
        )

    @property
    def file_path(self) -> Path:
        return self.file_paths[0]

    @property
    def mode(self) -> FileMode:
        return self.modes[0]

    @staticmethod
    def _initialize_h5_run(
        wepy_h5: WepyHDF5,
        init_walkers: list[Walker[WalkerState_]],
        resampling_fields: tuple[str, ...],
        decision_enum_dict: dict[str, int],
        continue_run: int | None = None,
        resampler_fields: tuple[str, ...] | None = None,
        warping_fields: tuple[str, ...] | None = None,
        progress_fields: tuple[str, ...] | None = None,
        bc_fields: tuple[str, ...] | None = None,
        resampling_records: tuple[str, ...] | None = None,
        resampler_records: tuple[str, ...] | None = None,
        bc_records: tuple[str, ...] | None = None,
        warping_records: tuple[str, ...] | None = None,
        progress_records: tuple[str, ...] | None = None,
    ) -> int:
        """Initialize the WepyHDF5 data structures."""

        if wepy_h5.mode != "r+":
            raise IOError(f"wepy_h5 must be in non-creation read-write mode (r+), in '{wepy_h5.mode}'")

        if not wepy_h5.closed:
            raise IOError("WepyHDF5 is already open, must be closed.")

        with wepy_h5:
            # if this is a continuation run of another run we want to
            # initialize it as such

            # initialize a new run, we don't know which run it will be
            # until it is created.
            run_grp = wepy_h5.new_run(
                init_walkers,
                continue_run=continue_run,
            )
            wepy_run_idx = run_grp.attrs["run_idx"]

            # initialize the run record groups using their fields
            wepy_h5.init_run_fields_resampling(
                wepy_run_idx,
                resampling_fields,
            )
            # the enumeration for the values of resampling
            wepy_h5.init_run_fields_resampling_decision(
                wepy_run_idx,
                decision_enum_dict,
            )

            if resampler_fields is not None:
                wepy_h5.init_run_fields_resampler(
                    wepy_run_idx,
                    resampler_fields,
                )
            # set the fields that are records for tables etc. unless
            # they are already set
            if resampling_records is not None and  "resampling" not in wepy_h5.record_fields:
                wepy_h5.init_record_fields(
                    "resampling",
                    resampling_records,
                )
            if resampler_records is not None and "resampler" not in wepy_h5.record_fields:
                wepy_h5.init_record_fields(
                    "resampler",
                    resampler_records,
                )

            # if there were no warping fields set there is no boundary
            # conditions and we don't initialize them
            if warping_fields is not None:
                wepy_h5.init_run_fields_warping(
                    wepy_run_idx,
                    warping_fields,
                )
                wepy_h5.init_run_fields_progress(
                    wepy_run_idx,
                    progress_fields,
                )
                wepy_h5.init_run_fields_bc(
                    wepy_run_idx,
                    bc_fields,
                )
                # table records
                if "warping" not in wepy_h5.record_fields:
                    wepy_h5.init_record_fields("warping", warping_records)
                if "boundary_conditions" not in wepy_h5.record_fields:
                    wepy_h5.init_record_fields(
                        "boundary_conditions", bc_records
                    )
                if "progress" not in wepy_h5.record_fields:
                    wepy_h5.init_record_fields("progress", progress_records)

        return wepy_run_idx

    @staticmethod
    def _resolve_state_units(
            units: dict[str, openmm.unit.Unit],
            state: WalkerStateBox,
    ) -> tuple[WalkerStateBox, dict[str, openmm.unit.Unit]]:
        """For walker states convert all quantity field values to plain values.

        Currently only supports openmm.unit.

        Returns the units used. If these were dynamically discovered
        from the quantity it will be that, otherwise it will be the
        unit that was passed in.

        """

        units_used = {}
        new_walker_fields = {}
        for field_key, field_value in state.dict().items():
            if not isinstance(field_value, Quantity):
                new_walker_fields[field_key] = field_value
            elif isinstance(field_value, openmm.unit.Quantity):

                # if there is a configured unit, convert to that
                if field_key in units:

                    unit = units[field_key]

                    units_used[field_key] = unit

                    new_walker_fields[field_key] = field_value.value_in_unit(unit)
                # If there is no unit for it, just get the
                # magnitude in the current units
                else:
                    new_walker_fields[field_key] = field_value.value_in_unit(field_value.unit)

                    units_used[field_key] = field_value.unit

        return WalkerStateBox(**new_walker_fields), units_used
        
    
    def init(self, **kwargs: SimComponentArgs) -> None:

        # TODO: remove dynamic configuration. Instead replace with
        # static configuration from the Runner for good defaults.
        
        ## Do checks on the inputs and figure out runtime field metadata

        # if we specify save fields only save these for the initial walkers
        if self.save_fields is not None:
            state_fields = list(kwargs["init_walkers"][0].state.dict().keys())

            # make sure all the save_fields are present in the state
            assert all(
                [
                    True if save_field in state_fields else False
                    for save_field in self.save_fields
                ]
            ), "Not all specified save_fields present in walker states"

            filtered_init_walkers = []
            for walker in kwargs["init_walkers"]:
                # make a new state by filtering the attributes of the old ones
                state_d = {
                    k: v
                    for k, v in walker.state.dict().items()
                    if k in self.save_fields
                }

                # and saving alternate representations as we would
                # expect them

                # if there are any alternate representations set them
                for alt_rep_name, alt_rep_idxs in self.alt_reps_idxs.items():
                    alt_rep_path = "alt_reps/{}".format(alt_rep_name)

                    # if the idxs are None we want all of the atoms
                    if alt_rep_idxs is None:
                        state_d[alt_rep_path] = state_d["positions"][:]
                    # otherwise get only the atoms we want
                    else:
                        state_d[alt_rep_path] = state_d["positions"][alt_rep_idxs]

                # always store a copy of the all_atoms rep for the init_walkers
                state_d[f"alt_reps/{self.ALL_ATOMS_REP_KEY}"] = state_d["positions"]

                # if the main rep is different then the full state
                # positions set that
                if self.main_rep_idxs is not None:
                    state_d["positions"] = state_d["positions"][self.main_rep_idxs]

                # TODO: reusing the state infrastructure here is not
                # the right thing. Currently just using a Box type to
                # get around this but it really should just be it's
                # own standalone type.

                # then making the new state
                new_state = WalkerStateBox(**state_d)

                filtered_init_walkers.append(Walker(new_state, walker.weight))
        # otherwise save the full state
        else:
            filtered_init_walkers = kwargs["init_walkers"]

        # If the state field values are quantities convert them to
        # plain values. 
        converted_filtered_init_walkers = []
        for walker_idx, init_walker in enumerate(filtered_init_walkers):
            _state, units_used = self._resolve_state_units(self.units, init_walker.state)

            # If no self.units were given, use the first
            # init walker to determine the units for a field overall, set
            # this and use for the rest of the walkers
            if walker_idx == 0:
                self.units.update(units_used)

            converted_filtered_init_walkers.append(
                Walker(state=_state, weight=init_walker.weight)
            )

        # Run the constructor intialization
        logger.info(f"Initializing HDF5 file at {self.file_path}")

        # convert units to strings
        _str_units = {
            key : str(unit)
            for key, unit
            in self.units.items()
        }

        init_wepy_h5 = WepyHDF5(
            self.file_path,
            mode="x",
            topology=self._tmp_topology,
            units=_str_units,
            sparse_fields=list(self._sparse_fields.keys()),
            feature_shapes_overrides=self._feature_shapes,
            feature_dtypes_overrides=self._feature_dtypes,
            n_dims=self._n_dims,
            main_rep_idxs=self.main_rep_idxs,
            alt_reps=self.alt_reps_idxs,
        )

        # delete the topology as it isn't needed anymore and we can
        # get it from the HDF5. This will alleviate some memory
        # pressure for large topologies
        del self._tmp_topology
        self._tmp_topology = None

        # then the file that is the actual attribute is opened in
        # read-write non-create mode
        self.wepy_h5 = WepyHDF5(
            self.file_path,
            mode='r+',
        )

        self.wepy_run_idx = self._initialize_h5_run(
            self.wepy_h5,
            init_walkers=converted_filtered_init_walkers,
            continue_run=kwargs["continue_run"],
            resampling_fields=self.resampling_fields,
            decision_enum_dict=self.decision_enum_dict,
            resampler_fields=self.resampler_fields,
            warping_fields=self.warping_fields,
            progress_fields=self.progress_fields,
            bc_fields=self.bc_fields,
            resampling_records=self.resampling_records,
            resampler_records=self.resampler_records,
            bc_records=self.bc_records,
            warping_records=self.warping_records,
            progress_records=self.progress_records,
        )

    def report(
        self,
        **kwargs: CycleReportDict,
    ) -> None:
        n_walkers = len(kwargs["new_walkers"])

        # determine which fields to save. If there were none specified
        # save all of them
        if self.save_fields is None:
            save_fields = list(kwargs["new_walkers"][0].state.dict().keys())
        else:
            save_fields = self.save_fields

        with self.wepy_h5:
            # turn on SWMR mode if requested for reporting
            if self.swmr_mode:
                self.wepy_h5.swmr_mode = True

            # add trajectory data for the walkers
            for walker_idx, walker in enumerate(kwargs["new_walkers"]):
                walker_weight = walker.weight
                walker_data = walker.state.dict()

                # iterate through the feature vectors of the walker
                # (fields), and the keys for the alt_reps
                for field_path in list(walker_data.keys()):
                    # save the field if it is in the list of save_fields
                    if field_path not in save_fields:
                        walker_data.pop(field_path)
                        continue

                    # if the result is None don't save anything
                    if walker_data[field_path] is None:
                        walker_data.pop(field_path)
                        continue

                    # if this is a sparse field we decide
                    # whether it is a valid cycle to save on
                    if field_path in self._sparse_fields:
                        if kwargs["cycle_idx"] % self._sparse_fields[field_path] != 0:
                            # this is not a valid cycle so we
                            # remove from the walker_data
                            walker_data.pop(field_path)
                            continue

                # Convert the walker data to plain non-quantity values. Only
                # do this for the save fields to avoid expensive conversions
                # for unused fields (e.g. forces)
                #
                # NOTE: do this before creating derived fields so the
                # Quantities don't propagate to them
                _state_noq, _units_used = self._resolve_state_units(
                    units=self.units,
                    state=WalkerStateBox(**walker_data),
                )
                _walker_data_noq = _state_noq.dict()

                # Add the alt_reps fields by slicing the positions
                for alt_rep_key in self.alt_reps_to_save:

                    alt_rep_idxs = self.alt_reps_idxs[alt_rep_key]
                    alt_rep_path = "alt_reps/{}".format(alt_rep_key)

                    # if the alt rep is also a sparse field check this
                    if alt_rep_path in self._sparse_fields:
                        # check to make sure this is a cycle this is
                        # to be saved to, if it is not continue on to
                        # the next field without saving this one
                        if kwargs["cycle_idx"] % self._sparse_fields[alt_rep_path] != 0:
                            continue

                    # slice them and save them

                    # if the idxs are None we want all of the atoms
                    if alt_rep_idxs is None:
                        alt_rep_data = _walker_data_noq["positions"][:]
                    # otherwise get only th atoms we want
                    else:
                        alt_rep_data = _walker_data_noq["positions"][alt_rep_idxs]

                    _walker_data_noq[alt_rep_path] = alt_rep_data

                # lastly reduce the atoms for the main representation
                # if this option was given
                if self.main_rep_idxs is not None:
                    _walker_data_noq["positions"] = _walker_data_noq["positions"][
                        self.main_rep_idxs
                    ]


                # for all of these fields we wrap them in additional
                # dimensions to make them feature vectors
                for field_path in list(walker_data.keys()):

                    # first if its a scalar wrap in the first layer
                    _val = _walker_data_noq[field_path]
                    if np.isscalar(_val):
                        _val = np.array([_val])

                    # then reshape to feature vector
                    _val = _val.reshape((1, *_val.shape))

                    _walker_data_noq[field_path] = _val

                # save the data to the HDF5 file for this walker

                # check to see if the walker has a trajectory in the run
                if walker_idx in self.wepy_h5.run_traj_idxs(self.wepy_run_idx):
                    # if it does then append to the trajectory
                    self.wepy_h5.extend_traj(
                        self.wepy_run_idx,
                        walker_idx,
                        weights=np.array([[walker_weight]]),
                        data=_walker_data_noq,
                    )
                # start a new trajectory
                else:
                    # add the traj for the walker with the data
                    traj_grp = self.wepy_h5.add_traj(
                        self.wepy_run_idx,
                        weights=np.array([[walker_weight]]),
                        data=_walker_data_noq,
                    )

                    # add as metadata the cycle idx where this walker started
                    traj_grp.attrs["cycle_idx"] = kwargs["cycle_idx"]

            # report the boundary conditions records data, if boundary
            # conditions were initialized
            if self.warping_fields is not None:
                self._report_warping(kwargs["cycle_idx"], kwargs["warp_data"])
                self._report_bc(kwargs["cycle_idx"], kwargs["bc_data"])
                self._report_progress(kwargs["cycle_idx"], kwargs["progress_data"])

            # report the resampling records data
            self._report_resampling(
                kwargs["cycle_idx"],
                kwargs["resampling_data"],
            )

            self._report_resampler(kwargs["cycle_idx"], kwargs["resampler_data"])


    def cleanup(self, **kwargs: SimComponentArgs) -> None:
        # # it should be already closed at this point but just in case
        # if not self.wepy_h5.closed:
        #     self.wepy_h5.close()

        # remove reference to the WepyHDF5 file so we can serialize this object
        del self.wepy_h5

        
    # sporadic
    def _report_warping(self, cycle_idx: int, warping_data: list[WarpingRecord_]) -> None:
        """Method to write warping specific information.

        Parameters
        ----------
        cycle_idx : int

        warp_data : list of dict of str : value
            List of dict-like records for each warping event from the
            last cycle.

        """

        if len(warping_data) > 0:
            self.wepy_h5.extend_cycle_warping_records(
                self.wepy_run_idx, cycle_idx, warping_data
            )

    def _report_bc(self, cycle_idx: int, bc_data: list[BCRecord_]) -> None:
        """Method to write boundary condition update specific information.

        Parameters
        ----------
        cycle_idx : int

        bc_data : list of dict of str : value
           List of dict-like records specifying the changes to the
           state of the boundary conditions in the last cycle.

        """

        if len(bc_data) > 0:
            self.wepy_h5.extend_cycle_bc_records(self.wepy_run_idx, cycle_idx, bc_data)

    def _report_resampler(
            self,
            cycle_idx: int,
            resampler_data: list[ResamplerRecord_],
    ) -> None:
        """Method to write resampler update specific information.

        Parameters
        ----------
        cycle_idx : int

        resampler_data : list of dict of str : value
            List of records specifying the changes to the state of the
            resampler in the last cycle.

        """

        if len(resampler_data) > 0:
            self.wepy_h5.extend_cycle_resampler_records(
                self.wepy_run_idx,
                cycle_idx,
                resampler_data,
            )

    # the resampling records are provided every cycle but they need to
    # be saved as sporadic because of the variable number of walkers
    def _report_resampling(
            self,
            cycle_idx: int,
            resampling_records: list[ResamplingRecord_],
    ) -> None:
        """Method to write resampling specific information.

        Parameters
        ----------
        cycle_idx : int

        resampling_data : list of dict of str : value
            List of records specifying the resampling to occur at this
            cycle.

        """

        self.wepy_h5.extend_cycle_resampling_records(
            self.wepy_run_idx, cycle_idx, resampling_records
        )

    # continual
    def _report_progress(self, cycle_idx: int, progress_data: ProgressRecord_) -> None:
        """Method to write progress specific information.

        Parameters
        ----------
        cycle_idx : int

        progress_data : dict str : list
            A record indicating the progress values for each walker in
            the last cycle.

        """

        self.wepy_h5.extend_cycle_progress_records(
            self.wepy_run_idx, cycle_idx, [progress_data]
        )
