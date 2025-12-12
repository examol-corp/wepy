import pytest
import numpy as np
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy_tools.systems.lennard_jones import LennardJonesPair
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.walker import Walker
from wepy.runners.mock import MockState, MockRunner
from wepy.work_mapper.serial import SerialMapper
from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NothingDecisionEnum,
)

SIM_COMPONENTS = {
    "init_walkers": [
        Walker(
            MockState(1),
            weight=0.2,
        ),
        Walker(
            MockState(1),
            weight=0.1,
        ),
    ],
    "runner": MockRunner(),
    "resampler": NoResampler(),
    "boundary_conditions": None,
    "work_mapper": SerialMapper(),
    "reporters": [],
    "continue_run": None,
}

RESAMPLER_REPORTER_ARGS = {
    "resampling_fields" : NoResampler.resampling_fields(),
    "decision_enum_dict" : NoDecision.enum_dict_by_name(),
}

class Test_WepyHDF5Reporter:

    def test___init__(self, tmp_path_factory):

        # TODO: should just use LJ pair for speed, save this for e2e
        # tests
        test_sys = LennardJonesPair()
        n_atoms = test_sys.mdj_top.n_atoms

        # NOTE: we don't need multiple of these because in this test
        # we don't actually write to a file yet
        d0 = tmp_path_factory.mktemp("0")

        h5_path = d0 / "main.wepy.h5"
        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
        )

        assert reporter.file_path == h5_path
        assert reporter.mode == "x"
        assert reporter.file_paths == [h5_path]
        assert reporter.modes == ["x"]

        assert not reporter.swmr_mode
        assert reporter.wepy_run_idx is None

        assert reporter._sparse_fields == {}
        assert reporter.alt_reps_to_save == []
        assert "all_atoms" in reporter.alt_reps_idxs
        assert np.array_equal(
            reporter.alt_reps_idxs["all_atoms"],
            np.array(range(n_atoms)),
        )
        assert reporter.main_rep_idxs is None

        assert reporter.units == {}

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions", "box_vectors",)
        )

        assert reporter.save_fields == ("positions", "box_vectors",)

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            units={
                "positions" : "nanometer",
            }
        )

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            main_rep_idxs=[0,1,2,3],
        )
        assert np.array_equal(
            reporter.main_rep_idxs,
            np.array([0,1,2,3])
        )
        assert set(reporter.alt_reps_idxs.keys()) == {"all_atoms"}
        assert "alt_reps/all_atoms" not in reporter.alt_reps_to_save
        assert "alt_reps/all_atoms" not in reporter._sparse_fields

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            all_atoms_rep_freq=5,
        )
        assert set(reporter.alt_reps_idxs.keys()) == {"all_atoms"}
        assert "all_atoms" in reporter.alt_reps_to_save
        assert reporter._sparse_fields == {
            "alt_reps/all_atoms" : 5
        }

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions", "velocities",),
            sparse_fields={
                "velocities" : 5,
            },
        )

        assert "velocities" in reporter._sparse_fields
        assert reporter._sparse_fields["velocities"] == 5

        with pytest.raises(ValueError):
            WepyHDF5Reporter(
                file_path=h5_path,
                topology=test_sys.json_top,
                **RESAMPLER_REPORTER_ARGS,
                save_fields=("positions",),
                sparse_fields={
                    "velocities" : 5,
                },
            )

        with pytest.raises(ValueError):
            WepyHDF5Reporter(
                file_path=h5_path,
                topology=test_sys.json_top,
                **RESAMPLER_REPORTER_ARGS,
                save_fields=None,
                sparse_fields={
                    "velocities" : 5,
                },
            )

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            alt_reps={
                "a" : (
                    [0,1,2,3],
                    5,
                ),
            }
        )

        assert "a" in reporter.alt_reps_to_save
        assert "a" in reporter.alt_reps_idxs
        assert np.array_equal(
            reporter.alt_reps_idxs["a"],
            np.array([0,1,2,3]),
        )
        assert "alt_reps/a" in reporter._sparse_fields
        assert reporter._sparse_fields["alt_reps/a"] == 5

        with pytest.raises(ValueError):
            WepyHDF5Reporter(
                file_path=h5_path,
                topology=test_sys.json_top,
                **RESAMPLER_REPORTER_ARGS,
                save_fields=("positions",),
                alt_reps={
                    "a" : (
                        [],
                        5,
                    ),
                }
            )

        with pytest.raises(ValueError):
            WepyHDF5Reporter(
                file_path=h5_path,
                topology=test_sys.json_top,
                **RESAMPLER_REPORTER_ARGS,
                save_fields=("positions",),
                alt_reps={
                    "a" : (
                        [0,1,2,3],
                        -1,
                    ),
                }
            )
            
        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            alt_reps={
                "a" : (
                    [0,1,2,3],
                    1,
                ),
            }
        )
        assert "a" in reporter.alt_reps_to_save
        assert "a" in reporter.alt_reps_idxs
        assert "alt_reps/a" not in reporter._sparse_fields

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            alt_reps={
                "a" : (
                    [0,1,2,3],
                    Ellipsis,
                ),
            }
        )
        assert "a" in reporter.alt_reps_to_save
        assert "a" in reporter.alt_reps_idxs
        assert "alt_reps/a" not in reporter._sparse_fields

    def test_from_components(self, tmp_path_factory):

        test_sys = LennardJonesPair()

        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        WepyHDF5Reporter.from_components(
            file_path=h5_path,
            topology=test_sys.json_top,
            resampler_class=NoResampler,
        )

    def test__initialize_h5_run(self, tmp_path_factory):
        test_sys = LennardJonesPair()

        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        # create the starting file
        WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        wepy_h5 = WepyHDF5(
            h5_path,
            mode="r+",
        )

        common_kwargs = dict(
            init_walkers=[
                Walker(
                    MockState(1),
                    0.1,
                ),
                Walker(
                    MockState(2),
                    0.2,
                ),
            ],
            continue_run=None,
        )

        run_idx = WepyHDF5Reporter._initialize_h5_run(
            wepy_h5=wepy_h5,
            **RESAMPLER_REPORTER_ARGS,
            **common_kwargs,
        )
        assert run_idx == 0

        wepy_h5_ro = WepyHDF5(
            h5_path,
            mode="r",
        )

        with pytest.raises(IOError):
            WepyHDF5Reporter._initialize_h5_run(
                wepy_h5=wepy_h5_ro,
                **RESAMPLER_REPORTER_ARGS,
                **common_kwargs,
            )

        # test that what we want is created

        wepy_h5_ro.open()
        assert str(run_idx) in wepy_h5_ro.h5["runs"]
        assert set(wepy_h5_ro.h5["runs/0"].keys()) == {
            "decision",
            "init_walkers",
            "resampling",
            "trajectories"
        }

        # decision
        assert "NOTHING" in wepy_h5_ro.h5["runs/0/decision"]
        assert wepy_h5_ro.h5["runs/0/decision/NOTHING"][()] == np.int64(0)

        # resampling
        assert set(wepy_h5_ro.h5["runs/0/resampling"].keys()) == {
            "_cycle_idxs",
            "decision_id",
            "step_idx",
            "target_idxs",
            "walker_idx",
        }

        # trajectories
        assert len(wepy_h5_ro.h5["runs/0/trajectories"].keys()) == 0

        # resampling records
        assert wepy_h5_ro.h5["runs/0/resampling/_cycle_idxs"].shape == (0,)
        assert wepy_h5_ro.h5["runs/0/resampling/decision_id"].shape == (0,1)
        assert wepy_h5_ro.h5["runs/0/resampling/step_idx"].shape == (0,1)
        assert wepy_h5_ro.h5["runs/0/resampling/walker_idx"].shape == (0,1)

        # init walkers
        assert len(wepy_h5_ro.h5["runs/0/init_walkers"]) == 2
        assert set(wepy_h5_ro.h5["runs/0/init_walkers"].keys()) == {"0", "1"}
        assert set(wepy_h5_ro.h5["runs/0/init_walkers/0"].keys()) == {"a", "weights"}

        # TOREV: weights are shape (1,1) don't see a good reason why,
        # but I think there was something about this. Perhaps instead
        # the 'a' field should be nested this way
        assert np.array_equal(
            wepy_h5_ro.h5["runs/0/init_walkers/0/a"][:],
            np.array([1])
        )
        assert np.array_equal(
            wepy_h5_ro.h5["runs/0/init_walkers/1/a"][:],
            np.array([2])
        )

        assert np.array_equal(
            wepy_h5_ro.h5["runs/0/init_walkers/0/weights"][:],
            np.array([[0.1]])
        )
        assert np.array_equal(
            wepy_h5_ro.h5["runs/0/init_walkers/1/weights"][:],
            np.array([[0.2]])
        )

        wepy_h5_ro.close()

        common_kwargs = dict(
            init_walkers=[
                Walker(
                    MockState(1),
                    0.1,
                ),
                Walker(
                    MockState(2),
                    0.2,
                ),
            ],
            continue_run=0,
        )
        run_idx = WepyHDF5Reporter._initialize_h5_run(
            wepy_h5=wepy_h5,
            **RESAMPLER_REPORTER_ARGS,
            **common_kwargs,
        )

        assert run_idx == 1
        wepy_h5_ro.open()
        assert "1" in wepy_h5_ro.h5["runs"]
        assert wepy_h5_ro.h5["_settings/continuations"].shape == (1, 2)
        assert np.array_equal(
            wepy_h5_ro.h5["_settings/continuations"][:],
            np.array([
                [1, 0]
            ])
        )



    def test_init(self, tmp_path_factory):

        test_sys = LennardJonesPair()
        n_atoms = test_sys.mdj_top.n_atoms

        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
        )

        reporter.init(**SIM_COMPONENTS)
