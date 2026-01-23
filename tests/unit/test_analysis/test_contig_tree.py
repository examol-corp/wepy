from pathlib import Path
import wepy
from wepy.analysis.contig_tree import (
    BaseContigTree,
    ContigTree,
    Contig,
)


class Test_BaseContigTree:

    def test___init__(self, alanine_dipeptide_revo_wepy_hdf5: Path):

        wepy_h5 = wepy.WepyHDF5(alanine_dipeptide_revo_wepy_hdf5, mode='r')

        bct = BaseContigTree(
            wepy_h5,
            decision_class=wepy.MultiCloneMergeDecision,
        )

        # TODO: test multiple runs/files

class Test_ContigTree:
    pass

class Test_Contig:
    pass
