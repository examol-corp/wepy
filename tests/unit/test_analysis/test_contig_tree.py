# Standard Library
from pathlib import Path

# Third Party Library
import networkx as nx

# First Party Library
import wepy
from wepy.analysis.contig_tree import (
    BaseContigTree,
    Contig,
    ContigTree,
)


class Test_BaseContigTree:

    def test___init__(self, wepy_h5_full_init: Path):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.continuations == {(1, 0)}
        assert bct.run_idxs == {0, 1}

        assert set(bct.graph.nodes) == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }

        assert set(bct.graph.edges) == {
            ((0, 1), (0, 0)),
            ((1, 0), (0, 1)),
            ((1, 1), (1, 0)),
        }

        for node_id in bct.graph.nodes:
            node = bct.graph.nodes[node_id]
            assert set(node.keys()) == {
                "resampling_steps",
                "parent_idxs",
                "discontinuities",
            }

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
            runs=[0],
        )

        assert bct.continuations == set()
        assert bct.run_idxs == {0}

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
            runs=[0, 1],
        )

        assert bct.continuations == {(1, 0)}
        assert bct.run_idxs == {0, 1}

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
            runs=[0, 1],
            continuations=[(1, 0)],
        )

        assert bct.continuations == {(1, 0)}
        assert bct.run_idxs == {0, 1}

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
            runs=[0, 1],
            # NOTE: manually overriding, even if incorrect
            continuations=[(0, 1)],
        )

        assert bct.continuations == {(0, 1)}
        assert bct.run_idxs == {0, 1}

    def test_contig_to_run_trace(self):

        assert BaseContigTree.contig_trace_to_run_trace(
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
            ],
        ) == [
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (1, 0, 1),
        ]

    def test_run_trace_to_contig_trace(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.run_trace_to_contig_trace(
            [
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                (1, 0, 1),
            ]
        ) == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
        ]

    def test_contig_cycle_idx(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.contig_cycle_idx(0, 0) == 0
        assert bct.contig_cycle_idx(0, 1) == 1
        assert bct.contig_cycle_idx(1, 0) == 2
        assert bct.contig_cycle_idx(1, 1) == 3

    def test_get_branch_trace(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.get_branch_trace(
            run_idx=0,
            cycle_idx=0,
            start_contig_idx=0,
        ) == [(0, 0)]

        assert bct.get_branch_trace(
            run_idx=0,
            cycle_idx=3,
            start_contig_idx=0,
        ) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

    def test_trace_parent_table(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.trace_parent_table(
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
            discontinuities=False,
        ) == [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]

        # TODO: discontinuities

    def test__tree_leaves(self):

        assert set(
            BaseContigTree._tree_leaves(
                (0, 0),
                # NOTE: the tree is reversed from what is in the BaseContigTree
                nx.DiGraph(
                    [
                        (
                            (0, 0),
                            (0, 1),
                        ),
                        (
                            (0, 1),
                            (1, 0),
                        ),
                        # leaves
                        (
                            (1, 0),
                            (1, 1),
                        ),
                        (
                            (1, 0),
                            (2, 0),
                        ),
                    ]
                ),
            )
        ) == {
            (1, 1),
            (2, 0),
        }

    def test__subtree_leaves(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert set(bct._subtree_leaves((0, 0))) == {(1, 1)}

    def test_leaves(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert set(bct.leaves()) == {(1, 1)}

    def test_root_leaves(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        root_leaves = bct.root_leaves()
        assert (0, 0) in root_leaves
        assert set(root_leaves[(0, 0)]) == {
            (1, 1),
        }

    def test_subtrees(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        # for one tree its the same
        assert len(bct.subtrees()) == 1

    def test_get_subtree(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        # TODO: this method is fallacious and need reworked, so just a
        # minimal test
        bct.get_subtree((0, 0)).nodes

    def test__subtree_root(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct._subtree_root((0, 0)) == (0, 0)
        assert bct._subtree_root((0, 1)) == (0, 0)
        assert bct._subtree_root((1, 0)) == (0, 0)
        assert bct._subtree_root((1, 1)) == (0, 0)

    def test_roots(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.roots() == [(0, 0)]

    def test_span_traces(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.span_traces == {
            0: [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
        }

    def test__root_spanning_contig_traces(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct._root_spanning_contig_traces() == {
            (0, 0): [
                [
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1),
                ],
            ],
        }

    def test_spanning_contig_traces(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.spanning_contig_traces() == [
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
        ]

    def test__spanning_paths(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct._spanning_paths((0, 0)) == {
            (1, 1): [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ]
        }

    def test__contig_trace_to_contig_runs(self):

        assert BaseContigTree._contig_trace_to_contig_runs(
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
        ) == [0, 1]

    def test__contig_runs_to_continuations(self):

        assert BaseContigTree._contig_runs_to_continuations([0, 1]) == [[1, 0]]

    def test__continuations_to_contig_runs(self):

        assert BaseContigTree._continuations_to_contig_runs([[1, 0]]) == [0, 1]

    # TODO: requires Contig
    def test_span_contig(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert False

        assert bct.span_traces == {0: []}


class Test_ContigTree:

    def test___init__(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        ct = ContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert ct.wepy_h5.closed

        assert ct.base_contigtree is not None

    def test_make_contig(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        ct = ContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        contig = ct.make_contig([(0, 0), (0, 1), (1, 0), (1, 1)])

    # TODO:
    # def test_resampling_trace(self, wepy_h5_full_init):
    #     wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode='r')

    #     ct = ContigTree(
    #         wepy_h5,
    #         decision_class=wepy.NoDecision,
    #     )

    def test_final_trace(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        ct = ContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert ct.final_trace() == [
            (1, 0, 1),
            (1, 1, 1),
        ]

    def test_lineages(self, wepy_h5_full_init):
        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        ct = ContigTree(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert ct.lineages(
            [
                (1, 0, 1),
            ],  # (1, 1, 1)],
            discontinuities=False,
        ) == [
            [
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                (1, 0, 1),
            ],
        ]


class Test_Contig:

    def test___init__(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        contig = Contig(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert contig.contig_trace == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

        assert contig.num_cycles == 4

    def test_walker_trace_to_run_trace(self, wepy_h5_full_init):

        wepy_h5 = wepy.WepyHDF5(wepy_h5_full_init, mode="r")

        bct = Contig(
            wepy_h5,
            decision_class=wepy.NoDecision,
        )

        assert bct.walker_trace_to_run_trace(
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
            ],
        ) == [
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (1, 0, 1),
        ]
