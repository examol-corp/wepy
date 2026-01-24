from wepy.hdf5 import WepyHDF5
import numpy as np


def test__wepy_h5_run_init(_wepy_h5_run_init):

    with WepyHDF5(_wepy_h5_run_init, mode='r') as wepy_h5:

        wepy_h5.run(0)

        assert wepy_h5.num_run_trajs(0) == 0

        wepy_h5.resampling_grp(0)
        wepy_h5.decision_grp(0)
        wepy_h5.warping_grp(0)
        wepy_h5.progress_grp(0)

        assert "resampling" in wepy_h5.record_fields
        assert wepy_h5.record_fields["resampling"] == [
                "decision_id",
                "target_idxs",
                "step_idx",
                "walker_idx",
            ]

        assert "warping" in wepy_h5.record_fields
        assert wepy_h5.record_fields["warping"] == [
                "walker_idx",
                "target_idx",
                "weight"
            ]

        assert "progress" in wepy_h5.record_fields
        assert wepy_h5.record_fields["progress"] == [
                "ensemble_average",
                "walker_distances",
            ]
        
# test that each test gets its own copy
def test_wepy_h5_run_init_1(wepy_h5_run_init):

    with WepyHDF5(wepy_h5_run_init, mode='r') as wepy_h5:
        assert "mutation_flag" not in wepy_h5.h5

    # mutate
    with WepyHDF5(wepy_h5_run_init, mode='r+') as wepy_h5:
        wepy_h5.h5["mutation_flag"] = np.array([0])

def test_wepy_h5_run_init_2(wepy_h5_run_init):

    with WepyHDF5(wepy_h5_run_init, mode='r') as wepy_h5:
        assert "mutation_flag" not in wepy_h5.h5

    # mutate
    with WepyHDF5(wepy_h5_run_init, mode='r+') as wepy_h5:
        wepy_h5.h5["mutation_flag"] = np.array([0])

def test__wepy_h5_traj_init(_wepy_h5_traj_init):

    with WepyHDF5(_wepy_h5_traj_init, mode='r') as wepy_h5:

        assert "velocities" in wepy_h5.sparse_fields
        
        wepy_h5.run(0)

        assert len(wepy_h5.resampling_records([0])) == 2
        assert len(wepy_h5.progress_records([0])) == 1
        
        assert wepy_h5.num_run_trajs(0) == 2
        
        for traj_idx in (0, 1):
            assert "weights" in wepy_h5.traj(0, traj_idx)
            assert "positions" in wepy_h5.traj(0, traj_idx)
            assert "box_vectors" in wepy_h5.traj(0, traj_idx)
            assert "kinetic_energy" in wepy_h5.traj(0, traj_idx)
            assert "velocities" in wepy_h5.traj(0, traj_idx)

            assert wepy_h5.traj_field_entity(0, traj_idx, "weights").shape == (1, 1)
            assert wepy_h5.traj_field_entity(0, traj_idx, "positions").shape == (1, 2, 3)
            assert wepy_h5.traj_field_entity(0, traj_idx, "box_vectors").shape == (1, 3, 3)
            assert wepy_h5.traj_field_entity(0, traj_idx, "kinetic_energy").shape == (1, 1)

            assert "data" in wepy_h5.traj_field_entity(0, traj_idx, "velocities")
            assert "_sparse_idxs" in wepy_h5.traj_field_entity(0, traj_idx, "velocities")

            assert wepy_h5.traj_field_entity(0, traj_idx, "velocities")["data"].shape == (0, 0, 0)
            assert wepy_h5.traj_field_entity(0, traj_idx, "velocities")["data"].maxshape == (None, 2, 3)

            assert wepy_h5.traj_field_entity(0, traj_idx, "velocities")["_sparse_idxs"].shape == (0,)

def test__wepy_h5_full_init(_wepy_h5_full_init):

    with WepyHDF5(_wepy_h5_full_init, mode='r') as wepy_h5:

        assert wepy_h5.num_runs == 2

        # run 0
        assert wepy_h5.num_traj_frames(0, 0) == 2
        assert wepy_h5.num_traj_frames(0, 1) == 2

        assert len(wepy_h5.resampling_records([0])) == 4
        assert len(wepy_h5.progress_records([0])) == 2

        # run 1
        assert len(wepy_h5.continuations) == 1
        assert tuple(wepy_h5.continuations[0]) == (1, 0)
        assert wepy_h5.num_traj_frames(1, 0) == 2
        assert wepy_h5.num_traj_frames(1, 1) == 2

        assert len(wepy_h5.resampling_records([1])) == 4
        assert len(wepy_h5.progress_records([1])) == 2
        
