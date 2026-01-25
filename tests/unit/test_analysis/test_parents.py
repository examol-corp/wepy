from typing import NamedTuple
from wepy.analysis.parents import (
    resampling_panel,
    parent_panel,
    net_parent_table,
    parent_table_discontinuities,
    parent_cycle_discontinuities,
    ancestors,
    sliding_window,
    ParentForest,
)
from wepy.storage.protocol import RunRecord
from wepy.resampling.decisions.no_decision import NoDecision

def test_resampling_panel():


    # # simple case
    # assert resampling_panel(
    #     [
    #         RunRecord(
    #             cycle_idx=0,
    #             record=dict(
    #                 step_idx=0,
    #                 walker_idx=0,
    #                 decision_id=0,
    #                 target_idxs=(0,),
    #             )
    #         ),
    #         RunRecord(
    #             cycle_idx=0,
    #             record=dict(
    #                 step_idx=0,
    #                 walker_idx=1,
    #                 decision_id=0,
    #                 target_idxs=(1,),
    #             )
    #         ),
    #     ]
    # ) == [
    #     # cycle 0
    #     [
    #         # step 0
    #         [
    #             # walker 0
    #             {
    #                 "decision_id" : 0,
    #                 "target_idxs" : (0,)
    #             },
    #             # walker 1
    #             {
    #                 "decision_id" : 0,
    #                 "target_idxs" : (1,)
    #             },
    #         ]
    #     ]
    # ]

    # TODO: failing
    # with multiple steps
    assert resampling_panel(
        [
            # step 0
            RunRecord(
                cycle_idx=0,
                record=dict(
                step_idx=0,
                walker_idx=0,
                decision_id=0,
                target_idxs=(0,),)
            ),
            RunRecord(
                cycle_idx=0,
                record=dict(
                step_idx=0,
                walker_idx=1,
                decision_id=0,
                target_idxs=(1,),)
            ),

            # step 1
            RunRecord(
                cycle_idx=0,
                record=dict(
                step_idx=1,
                walker_idx=0,
                decision_id=0,
                target_idxs=(1,),)
            ),
            RunRecord(
                cycle_idx=0,
                record=dict(
                step_idx=1,
                walker_idx=1,
                decision_id=0,
                target_idxs=(0,),)
            ),
        ]
    ) == [
        # cycle 0
        [
            # step 0
            [
                # walker 0
                {
                    "decision_id" : 0,
                    "target_idxs" : (0,)
                },
                # walker 1
                {
                    "decision_id" : 0,
                    "target_idxs" : (1,)
                },
            ],
            # step 1
            [
                # walker 0
                {
                    "decision_id" : 0,
                    "target_idxs" : (1,)
                },
                # walker 1
                {
                    "decision_id" : 0,
                    "target_idxs" : (0,)
                },
            ],
        ]
    ]
    
def test_parent_panel():

    assert parent_panel(
        NoDecision,
        [
            # cycle 0
            [
                # step 0
                [
                    # walker 0
                    {
                        "decision_id" : 0,
                        "target_idxs" : (0,)
                    },
                    # walker 1
                    {
                        "decision_id" : 0,
                        "target_idxs" : (1,)
                    }
                ],
            ],
        ]
    ) == [
        [
            [0, 1],
        ]
    ]

    assert parent_panel(
        NoDecision,
        [
            # cycle 0
            [
                # step 0
                [
                    # walker 0
                    {
                        "decision_id" : 0,
                        "target_idxs" : (0,)
                    },
                    # walker 1
                    {
                        "decision_id" : 0,
                        "target_idxs" : (1,)
                    }
                ],
                # step 1
                [
                    # walker 0
                    {
                        "decision_id" : 0,
                        "target_idxs" : (1,)
                    },
                    # walker 1
                    {
                        "decision_id" : 0,
                        "target_idxs" : (0,)
                    }
                ],
            ],
        ]
    ) == [
        [
            [0, 1],
            [1, 0],
        ]
    ]
    
def test_net_parent_table():

    net_parent_table([
        [
            [0, 1],
        ]
    ]) == [
        # cycle 0
        [
            0, 1
        ]
    ]
    

    net_parent_table([
        [
            [0, 1],
            [1, 0],
        ]
    ]) == [
        # cycle 0
        [
            1, 0
        ]
    ]

# TODO: tests for discontinuities
#
# def test_parent_table_discontinuities():
#     pass

# def test_parent_cycle_discontinuities():
#     pass

def test_ancestors():

    assert ancestors(
        [
            [0, 1],
        ],
        cycle_idx=0,
        walker_idx=0,
        ancestor_cycle=0,
    ) == [
        (0, 0),
    ]

    assert ancestors(
        [
            [0, 1],
            [0, 1],
        ],
        cycle_idx=1,
        walker_idx=0,
        ancestor_cycle=0,
    ) == [
        (0, 0),
        (0, 1),
    ]

    assert ancestors(
        [
            [0, 1],
            [0, 1],
        ],
        cycle_idx=1,
        walker_idx=0,
        ancestor_cycle=1,
    ) == [
        (0, 1),
    ]

    assert ancestors(
        [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        cycle_idx=3,
        walker_idx=0,
        ancestor_cycle=0,
    ) == [
        (1, 0),
        (1, 1),
        (1, 2),
        (0, 3),
    ]

    assert ancestors(
        [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        cycle_idx=3,
        walker_idx=1,
        ancestor_cycle=0,
    ) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
    ]
    
# TODO: test this
#
# def test_sliding_window():
#     pass

# TODO: need Contig for this to work
class Test_ParentForest:
    pass
