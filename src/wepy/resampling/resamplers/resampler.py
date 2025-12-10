# Standard Library
import logging
from typing import Any, Protocol, TypeVar, Generic, Literal, Union

# Standard Library
from warnings import warn

# Third Party Library
import numpy as np

# First Party Library
from wepy.resampling.decisions.decision import Decision, DecisionRecord
from wepy.walker import Walker, WalkerState

logger = logging.getLogger(__name__)

WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)


class ResamplerError(Exception):
    """Error raised when some constraint on resampling properties is
    violated.
    """

    pass

class Resampler(Protocol, Generic[WalkerState_]):

    DECISION: Decision
    CYCLE_FIELDS: tuple[str, ...]
    CYCLE_SHAPES: tuple[tuple[int, ...], ...]
    CYCLE_DTYPES: tuple[int | float, ...]
    CYCLE_RECORD_FIELDS: None | tuple[str, ...]
    RESAMPLING_FIELDS: tuple[str, ...]
    RESAMPLING_SHAPES: tuple[
        Union[
            tuple[int, ...],
            Literal[Ellipsis],
            None,
        ],
        ...
    ]
    RESAMPLING_DTYPES: tuple[
        Union[
            np.dtype,
            None
        ],
        ...
    ]
    RESAMPLING_RECORD_FIELDS: None | tuple[str, ...]
    RESAMPLER_FIELDS: tuple[str, ...]
    RESAMPLER_SHAPES: tuple[
        Union[
            tuple[int, ...],
            Literal[Ellipsis],
            None,
        ],
        ...
    ]

    RESAMPLER_DTYPES: tuple[
        Union[np.dtype, None], ...
    ]
    RESAMPLER_RECORD_FIELDS: None | tuple[str, ...]

    def resampling_fields(self) -> tuple[
            tuple[str, ...],
            tuple[
                Union[
                    tuple[int, ...],
                    Literal[Ellipsis],
                    None,
                ],
                ...
            ],
            tuple[
                Union[
                    np.dtype,
                    None
                ],
                ...
            ],
    ]:
        ...

    def resampling_record_field_names(self) -> None | tuple[str, ...]:
        ...

    def resampler_record_field_names(self) -> None | tuple[str, ...]:
        ...
        
    def resample(
            self,
            walkers: list[Walker[WalkerState_]]
    ) -> tuple[
        list[Walker[WalkerState_]],
        # TODO: better types for this
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        ...

class ResamplerABC(Resampler):
    """Abstract base class for implementing resamplers.

    All subclasses of Resampler must implement the 'resample' method.

    If extra reporting on resampling and resampler updates desired
    subclassed resamplers should update the following class constants
    for specifying the decision class and the names, shapes, data
    types, and table-like records for each record group:

    - DECISION
    - RESAMPLING_FIELDS
    - RESAMPLING_SHAPES
    - RESAMPLING_DTYPES
    - RESAMPLING_RECORD_FIELDS
    - RESAMPLER_FIELDS
    - RESAMPLER_SHAPES
    - RESAMPLER_DTYPES
    - RESAMPLER_RECORD_FIELDS

    The DECISION constant should be a
    wepy.resampling.decisions.decision.Decision subclass.

    This base class provides some hidden methods that are useful for
    various purposes.

    To help maintain constraints on the number of walkers in a
    simulation the constructor allows for setting of a minimum and/or
    maximum of walkers.

    These values are allowed to be either integers, None, or Ellipsis.

    Integers set hard values for the minimum and maximum values.

    None indicates that the min and max are unbounded. For the min
    this translates to a value of 1 since there must always be one
    walker.

    Ellipsis is an indicator to determine the minimum and maximum
    dependent on the number of walkers provided for resampling.

    If the max_num_walkers is Ellipsis and the number of walkers given
    is 10 then the maximum will be set to 10 for that
    resampling. Conversely, for if the minimum is Ellipsis.

    If both the min and max are set to Ellipsis then the number of
    walkers is always kept the same.

    Note that this does not implement any algorithm that actually
    decides how many walkers there will be but just checks that these
    constraints are met by those implementations in subclasses.

    To allow for this checking the '_resample_init' method should be
    called at the beginning of the 'resample' method and the
    '_resample_cleanup' should be called at the end of the 'resample'
    method.



    """

    DECISION: Decision = Decision
    """The decision class for this resampler."""

    CYCLE_FIELDS: tuple[str, ...] = (
        "step_idx",
        "walker_idx",
    )
    """The fields that get added to the decision record for all resampling
    records. This places a record within a single destructured listing
    of records for a single cycle of resampling using the step and
    walker index.
    """

    CYCLE_SHAPES: tuple[tuple[int, ...], ...] = (
        (1,),
        (1,),
    )
    """Data shapes of the cycle fields."""

    CYCLE_DTYPES: tuple[int | float] = (
        int,
        int,
    )
    """Data types of the cycle fields """

    CYCLE_RECORD_FIELDS: tuple[str, ...] = (
        "step_idx",
        "walker_idx",
    )
    """Optional, names of fields to be selected for truncated
    representation of the record group.
    """

    # data for resampling performed (continual)
    RESAMPLING_FIELDS = DECISION.FIELDS + CYCLE_FIELDS
    """String names of fields produced in this record group.

    Resampling records are typically used to report on the details of
    how walkers are resampled for a given resampling step.

    Warning
    -------

    This is a critical function of many other components of the wepy
    framework and probably shouldn't be altered by most developers.

    Thi is where the information about cloning and merging of walkers
    is given. Seeing as this is a most of the value proposition of
    wepy as a tool getting rid of it will render most of the framework
    useless.

    But sticking to the 'loosely coupled, tightly integrated' mantra
    you are free to modify these fields. This would be useful for
    implementing resampling strategies that do not follow basic
    cloning and merging. Just beware, that most of the lineage based
    analysis will be broken without implementing a new Decision class.

    """

    RESAMPLING_SHAPES = DECISION.SHAPES + CYCLE_SHAPES
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    RESAMPLING_DTYPES = DECISION.DTYPES + CYCLE_DTYPES
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + CYCLE_RECORD_FIELDS
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # changes to the state of the resampler (sporadic)
    RESAMPLER_FIELDS = ()
    """String names of fields produced in this record group.

    Resampler records are typically used to report on changes in the
    state of the resampler.

    Notes
    -----

    These fields are not critical to the proper functioning of the
    rest of the wepy framework and can be modified freely.

    However, reporters specific to this resampler probably will make
    use of these records.

    """

    RESAMPLER_SHAPES = ()
    """Numpy-style shapes of all fields produced in records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A tuple of ints that specify the shape of the field element
       array.

    B. Ellipsis, indicating that the field is variable length and
       limited to being a rank one array (e.g. (3,) or (1,)).

    C. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    Note that the shapes must be tuple and not simple integers for rank-1
    arrays.

    Option B will result in the special h5py datatype 'vlen' and
    should not be used for large datasets for efficiency reasons.

    """

    RESAMPLER_DTYPES = ()
    """Specifies the numpy dtypes to be used for records.

    There should be the same number of elements as there are in the
    corresponding 'FIELDS' class constant.

    Each entry should either be:

    A. A `numpy.dtype` object.

    D. None, indicating that the first instance of this field will not
       be known until runtime. Any field that is returned by a record
       producing method will automatically interpreted as None if not
       specified here.

    """

    RESAMPLER_RECORD_FIELDS = ()
    """Optional, names of fields to be selected for truncated
    representation of the record group.

    These entries should be strings that are previously contained in
    the 'FIELDS' class constant.

    While strictly no constraints on to which fields can be added here
    you should only choose those fields whose features could fit into
    a plaintext csv or similar format.

    """

    # valid debug modes
    DEBUG_MODES = (
        True,
        False,
    )

    def __init__(
        self,
        min_num_walkers: int | None | type(Ellipsis) = Ellipsis,
        max_num_walkers: int | None | type(Ellipsis) = Ellipsis,
        debug_mode: bool = False,
        **kwargs,
    ) -> None:
        """Constructor for Resampler class

        Parameters
        ----------
        min_num_walkers : int or None or Ellipsis
            The minimum number of walkers allowed to have. None is
            unbounded, and Ellipsis preserves whatever number of
            walkers were given as input as the minimum.

        max_num_walkers : int or None or Ellipsis
            The maximum number of walkers allowed to have. None is
            unbounded, and Ellipsis preserves whatever number of
            walkers were given as input as the maximum.

        debug_mode : bool
            Expert mode stuff don't use unless you know what you are doing.


        """

        # The min and max number of walkers that can be generated in
        # resampling.

        # Ellipsis means to keep bound it by the number of
        # walkers given to the resample method (e.g. if
        # max_num_walkers == Ellipsis and min_num_walkers == 5 and
        # resample is given 10 then the max will be set to 10 for that
        # resampling and the min will always be 5. If both are
        # Ellipsis then the number of walkers is kept the same)

        # None means that there is no bound, e.g. max_num_walkers ==
        # None then there is no maximum number of walkers, however a
        # min_num_walkers of None in practice is 1 since there must
        # always be at least 1 walker

        if min_num_walkers not in {Ellipsis, None}:
            if min_num_walkers < 1:
                raise ResamplerError(
                    "The minimum number of walkers should be at least 1"
                )

            if max_num_walkers not in {Ellipsis, None} and  min_num_walkers > max_num_walkers:
                raise ResamplerError(
                    f"min_num_walkers ({min_num_walkers}) must be less than or equal to max_num_walkers ({max_num_walkers})"
                )

        self._min_num_walkers = min_num_walkers
        self._max_num_walkers = max_num_walkers

        # this will be used to save the number of walkers given during
        # resampling, we initialize to None
        self._resampling_num_walkers = None

        # initialize debug mode
        self._debug_mode = False

        # set them to the args given
        self.set_debug_mode(debug_mode)

    @property
    def decision(self) -> Decision:
        """The decision class for this resampler."""
        return self.DECISION

    def resampling_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.RESAMPLING_FIELDS

    def resampling_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.RESAMPLING_SHAPES

    def resampling_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.RESAMPLING_DTYPES

    def resampling_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------
        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(
            zip(
                self.resampling_field_names(),
                self.resampling_field_shapes(),
                self.resampling_field_dtypes(),
            )
        )

    def resampling_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.RESAMPLING_RECORD_FIELDS

    def resampler_field_names(self):
        """Access the class level FIELDS constant for this record group."""
        return self.RESAMPLER_FIELDS

    def resampler_field_shapes(self):
        """Access the class level SHAPES constant for this record group."""
        return self.RESAMPLER_SHAPES

    def resampler_field_dtypes(self):
        """Access the class level DTYPES constant for this record group."""
        return self.RESAMPLER_DTYPES

    def resampler_fields(self):
        """Returns a list of zipped field specs.

        Returns
        -------
        record_specs : list of tuple
            A list of the specs for each field, a spec is a tuple of
            type (field_name, shape_spec, dtype_spec)
        """
        return list(
            zip(
                self.resampler_field_names(),
                self.resampler_field_shapes(),
                self.resampler_field_dtypes(),
            )
        )

    def resampler_record_field_names(self):
        """Access the class level RECORD_FIELDS constant for this record group."""
        return self.RESAMPLER_RECORD_FIELDS

    @property
    def is_debug_on(self) -> bool:
        """ """
        return self._debug_mode

    def set_debug_mode(self, mode: bool) -> None:
        """Parameters
        ----------
        mode

        Returns
        -------

        """

        if mode not in self.DEBUG_MODES:
            raise ValueError("debug mode, {}, not valid".format(mode))

        self._debug_mode = mode

        # if you want to use debug mode you have to have ipdb installed
        if self.is_debug_on:
            try:
                # Third Party Library
                import ipdb
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "You must have ipdb installed to use the debug feature"
                )

    def debug_on(self) -> None:
        """ """
        if self.is_debug_on:
            warn("Debug mode is already on")

        self.set_debug_mode(True)

    def debug_off(self) -> None:
        """ """
        if not self.is_debug_on:
            warn("Debug mode is already off")

        self.set_debug_mode(False)

    @property
    def max_num_walkers_setting(self) -> int:
        """The specification for the maximum number of walkers for the resampler."""
        return self._max_num_walkers

    @property
    def min_num_walkers_setting(self) -> int:
        """The specification for the minimum number of walkers for the resampler."""
        return self._min_num_walkers

    def max_num_walkers(self) -> int | None:
        """ " Get the max number of walkers allowed currently"""

        # first check to make sure that a resampling is occuring and
        # we have a number of walkers to even reference
        if self._resampling_num_walkers is None:
            raise ResamplerError(
                "A resampling is currently not taking place so the"
                " current number of walkers is not known."
            )

        # we are in a resampling so there is a current value for the
        # max number of walkers
        else:
            # if the max is None then there is no max number of
            # walkers so we just return None
            if self.max_num_walkers_setting is None:
                return None

            # if the max is Ellipsis then we just return what the
            # current number of walkers is
            elif self.max_num_walkers_setting is Ellipsis:
                return self._resampling_num_walkers

            # if it is not those then it is a hard number and we just
            # return it
            else:
                return self.max_num_walkers_setting

    def min_num_walkers(self) -> int | None:
        """ " Get the min number of walkers allowed currently"""

        # first check to make sure that a resampling is occuring and
        # we have a number of walkers to even reference
        if self._resampling_num_walkers is None:
            raise ResamplerError(
                "A resampling is currently not taking place so the"
                " current number of walkers is not known."
            )

        # we are in a resampling so there is a current value for the
        # min number of walkers
        else:
            # if the min is None then there is no min number of
            # walkers so we just return None
            if self.min_num_walkers_setting is None:
                return None

            # if the min is Ellipsis then we just return what the
            # current number of walkers is
            elif self.min_num_walkers_setting is Ellipsis:
                return self._resampling_num_walkers

            # if it is not those then it is a hard number and we just
            # return it
            else:
                return self.min_num_walkers_setting

    def _set_resampling_num_walkers(self, num_walkers: int) -> None:
        """Sets the concrete number of walkers constraints given a number of
        walkers and the settings for max and min.

        Parameters
        ----------
        num_walkers : int

        """

        # there must be at least 1 walker in order to do resampling
        if num_walkers < 1:
            raise ResamplerError("No walkers were given to resample")

        # if the min number of walkers is not dynamic check to see if
        # this number violates the hard boundary
        if self._min_num_walkers in (None, Ellipsis):
            self._resampling_num_walkers = num_walkers
        elif num_walkers < self._min_num_walkers:
            raise ResamplerError(
                "The number of walkers given to resample is less than the minimum"
            )

        # if the max number of walkers is not dynamic check to see if
        # this number violates the hard boundary
        if self._max_num_walkers in (None, Ellipsis):
            self._resampling_num_walkers = num_walkers
        elif num_walkers < self._max_num_walkers:
            raise ResamplerError(
                "The number of walkers given to resample is less than the maximum"
            )

    def _unset_resampling_num_walkers(self) -> None:
        self._resampling_num_walkers = None

    def _resample_init(
        self,
        walkers: list[Walker[WalkerState_]],
    ) -> None:
        """Common initialization stuff for resamplers.

        Sets the number of walkers in this round of resampling.

        Parameters
        ----------
        walkers : list of Walker objects

        """

        # first set how many walkers there are in this resampling
        self._set_resampling_num_walkers(len(walkers))

    def _resample_cleanup(self, **kwargs) -> None:
        """Common cleanup stuff for resamplers.

        Unsets the number of walkers for this round of resampling.

        """

        # unset the number of walkers for this resampling
        self._unset_resampling_num_walkers()

    def resample(
        self,
        walkers: list[Walker[WalkerState_]],
        debug_mode: bool = False,
    ) -> tuple[
        list[Walker[WalkerState_]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Perform resampling on the set of walkers.

        Parameters
        ----------
        walkers : list of Walker objects
            The walkers that are to be resampled.

        debug_mode : bool
            Expert mode debugging setting, only forif you know exactly
            what you are doing.

        Returns
        -------
        resampled_walkers : list of Walker objects
            The set of resampled walkers

        resampling_data : list of dict of str: value
            A list of destructured resampling records from this round
            of resampling.

        resampler_data : list of dict of str: value
            A list of records recording how the state of the resampler
            was updated.

        """

        raise NotImplementedError
