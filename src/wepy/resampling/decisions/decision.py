"""Abstract base class for Decision classes.

See the NoDecision class and others in this module for examples.

To create your own subclass of the Decision class you must customize
the following class constants:

- ENUM
- FIELDS
- SHAPES
- DTYPE
- RECORD_FIELDS

The 'ENUM' field should be a python 'Enum' class created by
subclassing and customizing 'Enum' in the normal pythonic way. The
elements of the 'Enum' are the actual decision choices, and their
numeric value is used for serialization.

The 'FIELDS' constant is a specification of the number of fields that
a decision record will have. All decision records should contain the
'decision_id' field which is the choice of decision. This class
implements that and should be used as shown in the examples.

In order that fields be serializable to different formats, we also
require that they be a numpy array datatype.

To support this we require the data shapes and data types for each
field. Elements of SHAPES and DTYPES should be of a format
recognizable by the numpy array constructor.

To this we allow the additional option of specifying SHAPES as the
python built-in name Ellipsis (AKA '...'). This will specify the shape
as a variable length 1-dimensional array.

The RECORD_FIELDS is used as a way to specify fields which are
amenable to placement in simplified summary tables, i.e. simple
non-compound values.

The only method that needs to be implemented in the Decision is
'action'.

This function actually implements the algorithm for taking actions on
the decisions and instructions and is called from the resampler to
perform them on the collection of walkers.

"""

# Standard Library
from typing import TypedDict, Required, Any, Union
import logging

# Standard Library
from enum import IntEnum

import attrs

from wepy.walker import Walker

logger = logging.getLogger(__name__)

@attrs.define
class DecisionRecord:
    decision_id: int


DecisionFieldDtype = Union[int,]


# ABC for the Decision class
class Decision:
    """Represents and provides methods for a set of decision values."""

    ENUM: IntEnum
    """The enumeration of the decision types. Maps them to integers."""

    DEFAULT_DECISION: int
    """The default decision to choose."""

    FIELDS: tuple[str, ...] = ("decision_id",)
    """The names of the fields that go into the decision record."""

    # suggestion for subclassing, FIELDS and others
    # FIELDS = super().FIELDS + ('target_idxs',)
    # etc.

    #  An Ellipsis instead of fields indicate there is a variable
    # number of fields.
    SHAPES: tuple[tuple[int | type(Ellipsis), ...], ...] = ((1,),)
    """Field data shapes."""

    DTYPES: tuple[DecisionFieldDtype, ...] = (int,)
    """Field data types."""

    RECORD_FIELDS: tuple[str] = ("decision_id",)
    """The fields that could be used in a reduced table-like representation."""

    ANCESTOR_DECISION_IDS: tuple[int, ...]
    """Specify the enum values where their walker state sample value is
    passed on in the next generation, i.e. after performing the action."""

    @classmethod
    def default_decision(cls):
        return cls.DEFAULT_DECISION

    @classmethod
    def field_names(cls):
        """Names of the decision record fields."""
        return cls.FIELDS

    @classmethod
    def field_shapes(cls):
        """Field data shapes."""
        return cls.SHAPES

    @classmethod
    def field_dtypes(cls):
        """Field data types."""
        return cls.DTYPES

    @classmethod
    def fields(cls):
        """Specs for each field.

        Returns
        -------
        fields : list of tuples
            Field specs each spec is of the form (name, shape, dtype).

        """
        return list(zip(cls.field_names(), cls.field_shapes(), cls.field_dtypes()))

    @classmethod
    def record_field_names(cls):
        """The fields that could be used in a reduced table-like representation."""
        return cls.RECORD_FIELDS

    @classmethod
    def enum_dict_by_name(cls):
        """Get the decision enumeration as a dict mapping name to integer."""
        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.name] = enum.value
        return d

    @classmethod
    def enum_dict_by_value(cls):
        """Get the decision enumeration as a dict mapping integer to name."""

        if cls.ENUM is None:
            raise NotImplementedError

        d = {}
        for enum in cls.ENUM:
            d[enum.value] = enum
        return d

    @classmethod
    def enum_by_value(cls, enum_value):
        """Get the enum name for an enum_value.

        Parameters
        ----------
        enum_value : int

        Returns
        -------
        enum_name : enum

        """
        d = cls.enum_dict_by_value()
        return d[enum_value]

    @classmethod
    def enum_by_name(cls, enum_name):
        """Get the enum name for an enum_value.

        Parameters
        ----------
        enum_name : enum

        Returns
        -------
        enum_value : int

        """

        d = cls.enum_dict_by_name()
        return d[enum_name]

    @classmethod
    def record(cls, enum_value: int, **fields: dict[str, Any]) -> DecisionRecord:
        """Generate a record for the enum_value and the other fields.

        Parameters
        ----------
        enum_value : int

        Returns
        -------
        rec : dict of str: value

        """

        assert (
            enum_value in cls.enum_dict_by_value()
        ), "value is not a valid Enumerated value"

        for field_key in fields.keys():
            assert (
                field_key in cls.FIELDS
            ), "The field {} is not a field for that decision".format(field_key)
            assert field_key != "decision_id", "'decision_id' cannot be an extra field"

        rec = {"decision_id": enum_value}
        rec.update(fields)

        return rec

    @classmethod
    def action(
        cls,
        walkers: list[Walker],
        decisions: list[list[DecisionRecord]],
    ) -> list[Walker]:
        """Perform the instructions for a set of resampling records on
        walkers.

        The decisions are a collection of decision records which
        contain the decision value and the instruction values for a
        particular walker within its cohort (sample set).

        The collection is organized as a list of lists. The outer list
        corresponds to the steps of resampling within a cycle.

        The inner list is a list of decision records for a specific
        step of resampling, where the index of the decision record is
        the walker index.

        Parameters
        ----------
        walkers : list of Walker objects
            The walkers you want to perform the decision instructions on.

        decisions : list of list of decision records
            The decisions for each resampling step and their
            instructions to apply to the walkers.

        Returns
        -------
        resampled_walkers : list of Walker objects
            The resampled walkers.

        Raises
        ------
        NotImplementedError : abstract method

        """
        raise NotImplementedError

