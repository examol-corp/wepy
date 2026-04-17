# Standard Library
from enum import IntEnum

# Third Party Library
import pytest

# First Party Library
from wepy.resampling.decisions.decision import BaseDecisionABC, BaseDecisionRecord
from wepy.runners.mock import MockState
from wepy.walker import Walker


# minimal implementation of the ABC for testing
class MockDecisionEnum(IntEnum):
    NOTHING = 0


class MockDecision(BaseDecisionABC):

    ENUM = MockDecisionEnum
    DEFAULT_DECISION = ENUM.NOTHING
    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,)


class Test_BaseDecisionRecord:

    def test_to_dict(self):
        assert BaseDecisionRecord(decision_id=1, target_idxs=(0,)).to_dict() == {
            "decision_id": 1,
            "target_idxs": (0,),
        }


class Test_Decision:

    def test_default_decision(self):
        assert MockDecision.default_decision() == MockDecisionEnum.NOTHING

    def test_field_names(self):
        assert MockDecision.field_names() == (
            "decision_id",
            "target_idxs",
        )

    def test_field_shapes(self):
        assert MockDecision.field_shapes() == ((1,), Ellipsis)

    def test_field_dtypes(self):
        assert MockDecision.field_dtypes() == (int, int)

    def test_fields(self):
        assert MockDecision.fields() == [
            (
                "decision_id",
                (1,),
                int,
            ),
            (
                "target_idxs",
                Ellipsis,
                int,
            ),
        ]

    def test_record_field_names(self):
        assert MockDecision.record_field_names() == (
            "decision_id",
            "target_idxs",
        )

    def test_enum_dict_by_name(self):
        assert MockDecision.enum_dict_by_name() == {
            "NOTHING": 0,
        }

    def test_enum_dict_by_value(self):
        assert MockDecision.enum_dict_by_value() == {
            0: MockDecisionEnum.NOTHING,
        }

    def test_enum_by_value(self):
        assert MockDecision.enum_by_value(0) == MockDecisionEnum.NOTHING

    def test_enum_by_name(self):
        assert MockDecision.enum_by_name("NOTHING") == MockDecisionEnum.NOTHING

    def test_record(self):

        assert MockDecision.record(0, target_idxs=(0,)) == BaseDecisionRecord(
            decision_id=0,
            target_idxs=(0,),
        )

    def test_action(self):

        with pytest.raises(NotImplementedError):
            MockDecision.action(
                [Walker(MockState(1), 0.1) for _ in range(4)],
                [
                    BaseDecisionRecord(decision_id=0, target_idxs=(idx,))
                    for idx in range(4)
                ],
            )
