import pytest
from typing import TypedDict
from enum import IntEnum
import attrs
from wepy.resampling.decisions.decision import Decision, DecisionRecord
from wepy.walker import Walker
from wepy.runners.mock import MockState

# minimal implementation of the ABC for testing
class MockDecisionEnum(IntEnum):
    NOTHING = 0

class MockDecision(Decision):

    ENUM = MockDecisionEnum
    DEFAULT_DECISION = ENUM.NOTHING
    ANCESTOR_DECISION_IDS = (ENUM.NOTHING.value,)

class Test_Decision:

    def test_default_decision(self):
        assert MockDecision.default_decision() == MockDecisionEnum.NOTHING

    def test_field_names(self):
        assert MockDecision.field_names() == ("decision_id",)

    def test_field_shapes(self):
        assert MockDecision.field_shapes() == ((1,),)

    def test_field_dtypes(self):
        assert MockDecision.field_dtypes() == (int,)

    def test_fields(self):
        assert MockDecision.fields() == [
            ("decision_id", (1,), int,)
        ]
    def test_record_field_names(self):
        assert MockDecision.record_field_names() == ("decision_id",)

    def test_enum_dict_by_name(self):
        assert MockDecision.enum_dict_by_name() == {
            "NOTHING" : 0,
        }

    def test_enum_dict_by_value(self):
        assert MockDecision.enum_dict_by_value() == {
            0 : MockDecisionEnum.NOTHING,
        }

    def test_enum_by_value(self):
        assert MockDecision.enum_by_value(0) == MockDecisionEnum.NOTHING
    def test_enum_by_name(self):
        assert MockDecision.enum_by_name("NOTHING") == MockDecisionEnum.NOTHING

    def test_record(self):

        assert MockDecision.record(0) == DecisionRecord(decision_id=0)

    def test_action(self):

        with pytest.raises(NotImplementedError):
            MockDecision.action(
                [
                    Walker(
                        MockState(1),
                        0.1
                    )
                    for _ in range(4)
                ],
                [
                    DecisionRecord(
                        decision_id=0
                    )
                    for _ in range(4)
                ],
            )

