import pytest
import numpy as np
import openmm.unit
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy_tools.systems.lennard_jones import LennardJonesPair
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.walker import Walker, WalkerStateBox
from wepy.runners.mock import MockState, MockRunner
from wepy.runners.openmm import OpenMMState, OPENMM_DEFAULT_UNITS
from wepy.work_mapper.serial import SerialMapper
from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.no_decision import (
    NoDecision,
    NothingDecisionEnum,
)

LJ_OPENMM_SIM_COMPONENTS = {
    "init_walkers": [
                Walker(
                    OpenMMState.from_dwim(
                        positions=np.array([
                            [0., 0., 0.,],
                            [0., 0., 0.,],
                        ]) * openmm.unit.nanometer,
                        time=(1.3 * openmm.unit.nanosecond),
                    ),
                    0.5,
                ),
                Walker(
                    OpenMMState.from_dwim(
                        positions=np.array([
                            [0., 0., 0.,],
                            [0., 0., 0.,],
                        ]) * openmm.unit.nanometer,
                        time=(1.3 * openmm.unit.nanosecond),
                    ),
                    0.5,
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

CYCLE_REPORT_DICT_COMMON = {
            "runner_precycle_time" : 1.,
            "runner_postcycle_time" : 1.,
            "sim_manager_segment_overhead_time" : 1.,
            "cycle_sim_manager_segment_time" : 1.,
            "cycle_runner_time" : 1.,
            "cycle_bc_time" : 1.,
            "cycle_resampling_time" : 1.,
}

CYCLE_REPORT_DICT_EMPTY_OPTIONALS = {
            "warp_data" : [],
            "bc_data" : [],
            "progress_data" : [],
            "resampler_data" : [],
            "runner_splits_time" : None,
            "worker_segment_times" : None,
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

        assert reporter.units == OPENMM_DEFAULT_UNITS

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
                "positions" : openmm.unit.angstrom,
            }
        )

        assert reporter.units == dict(OPENMM_DEFAULT_UNITS) | {"positions" : openmm.unit.angstrom}

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

    def test__resolve_state_units(self):

        state_nm = WalkerStateBox(
            positions=np.array([
                [1., 1., 1.,],
                [1., 1., 1.,],
            ]) * openmm.unit.nanometer
        )

        state_nm_mags, units_used = WepyHDF5Reporter._resolve_state_units(
            units={
                "positions" : openmm.unit.nanometer,
            },
            state=state_nm,
        )
        assert units_used == {"positions" : openmm.unit.nanometer}
        assert isinstance(state_nm_mags["positions"], np.ndarray)
        assert np.array_equal(
            state_nm_mags["positions"],
            np.array([
                [1., 1., 1.,],
                [1., 1., 1.,],
            ]),
        )

        state_nm_mags, units_used = WepyHDF5Reporter._resolve_state_units(
            units={
                "positions" : openmm.unit.angstrom,
            },
            state=state_nm,
        )
        assert units_used == {"positions" : openmm.unit.angstrom}
        assert isinstance(state_nm_mags["positions"], np.ndarray)
        assert np.array_equal(
            state_nm_mags["positions"],
            np.array([
                [10., 10., 10.,],
                [10., 10., 10.,],
            ]),
        )

        # if no units specified the ones from the state
        state_nm_mags, units_used = WepyHDF5Reporter._resolve_state_units(
            units={},
            state=state_nm,
        )
        assert units_used == {"positions" : openmm.unit.nanometer}
        
    def test_init(self, tmp_path_factory):

        test_sys = LennardJonesPair()

        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units=None,
            **RESAMPLER_REPORTER_ARGS,
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        assert reporter.units == OPENMM_DEFAULT_UNITS

        assert reporter.wepy_run_idx == 0
        assert reporter._tmp_topology is None
        assert reporter.file_path == h5_path
        assert h5_path.exists()
        assert reporter.wepy_h5.mode == "r+"

        # minimal tests, see _initialize_h5_run for more in depth tests
        with reporter.wepy_h5 as wepy_h5:

            # should be defaults
            assert wepy_h5.h5["units/positions"][()].decode() == "nanometer"
            assert wepy_h5.h5["units/box_vectors"][()].decode() == "nanometer"
            assert wepy_h5.h5["units/box_volume"][()].decode() == "nanometer**3"
            assert wepy_h5.h5["units/time"][()].decode() == "picosecond"

            assert "0" in wepy_h5.h5["runs"]
            assert "init_walkers" in wepy_h5.h5["runs/0"]
            assert len(wepy_h5.h5["runs/0/init_walkers"]) == 2
            assert "0" in wepy_h5.h5["runs/0/init_walkers"]
            # all the fields in the state will be saved, since no save_fields given
            assert "positions" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "box_vectors" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "box_volume" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "time" in wepy_h5.h5["runs/0/init_walkers/0"]

            # compare to the different unit output later
            nanometer_bvs = wepy_h5.h5["runs/0/init_walkers/0/box_vectors"][:]

        # if no units are given, derive them dynamically from
        # quantities
        d1 = tmp_path_factory.mktemp("1")
        h5_path = d1 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units={
                # provide explicit units for all the encountered
                # fields. These should be the reporter units
                "box_vectors" : openmm.unit.angstrom,
                "time" : openmm.unit.nanosecond,
            },
            **RESAMPLER_REPORTER_ARGS,
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        assert {
            unit_name : unit
            for unit_name, unit
            in reporter.units.items()
            if unit_name in {"box_vectors", "time"}
        } == {
            "box_vectors" : openmm.unit.angstrom,
            "time" : openmm.unit.nanosecond,
        }

        assert {
            unit_name : unit
            for unit_name, unit
            in reporter.units.items()
            if unit_name not in {"box_vectors", "time"}
        } == {
            unit_name : unit
            for unit_name, unit
            in OPENMM_DEFAULT_UNITS.items()
            if unit_name not in {"box_vectors", "time"}
        }
        
        with reporter.wepy_h5 as wepy_h5:
            # the overridden ones
            assert wepy_h5.h5["units/box_vectors"][()].decode() == "angstrom"
            assert wepy_h5.h5["units/time"][()].decode() == "nanosecond"
            # some of the defaults
            assert wepy_h5.h5["units/positions"][()].decode() == "nanometer"
            assert wepy_h5.h5["units/box_volume"][()].decode() == "nanometer**3"

            # compare the numbers from each to make sure they have the same magnitude
            angstrom_bvs = wepy_h5.h5["runs/0/init_walkers/0/box_vectors"][:]
            assert np.array_equal(
                nanometer_bvs * 10,
                angstrom_bvs,
            )
        
        # test the init_walker_save_fields behavior
        d2 = tmp_path_factory.mktemp("2")
        h5_path = d2 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units=None,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            init_walker_save_fields=None,
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        with reporter.wepy_h5 as wepy_h5:
            assert "0" in wepy_h5.h5["runs"]
            assert "init_walkers" in wepy_h5.h5["runs/0"]
            assert len(wepy_h5.h5["runs/0/init_walkers"]) == 2
            assert "0" in wepy_h5.h5["runs/0/init_walkers"]
            assert "positions" in wepy_h5.h5["runs/0/init_walkers/0"]

            # the remainder of the fields that were in the state should not be saved
            assert not "box_vectors" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "box_volume" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "time" in wepy_h5.h5["runs/0/init_walkers/0"]
            

        ## Test the save fields family of arguments
        d3 = tmp_path_factory.mktemp("3")
        h5_path = d3 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units=None,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            init_walker_save_fields=("positions",),
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        with reporter.wepy_h5 as wepy_h5:
            assert "positions" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "box_vectors" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "box_volume" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "time" in wepy_h5.h5["runs/0/init_walkers/0"]

        # special cases for the init walkers, save all fields that
        # were given to it
        d4 = tmp_path_factory.mktemp("4")
        h5_path = d4 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units=None,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions",),
            init_walker_save_fields=Ellipsis,
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        with reporter.wepy_h5 as wepy_h5:
            assert "positions" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "box_vectors" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "box_volume" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "time" in wepy_h5.h5["runs/0/init_walkers/0"]

        # Match a different set of fields for init_walkers than the
        # save_fields
        d5 = tmp_path_factory.mktemp("5")
        h5_path = d5 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            units=None,
            **RESAMPLER_REPORTER_ARGS,
            save_fields=("positions", "box_vectors", "time",),
            # only take positions and box_vectors
            init_walker_save_fields=("positions", "box_vectors",),
        )

        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        with reporter.wepy_h5 as wepy_h5:
            assert "positions" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert "box_vectors" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "box_volume" in wepy_h5.h5["runs/0/init_walkers/0"]
            assert not "time" in wepy_h5.h5["runs/0/init_walkers/0"]
            
    def test_report(self, tmp_path_factory):

        # TODO: using the OpenMM Runner OpenMMState here because the
        # HDF5 requires a 'positions' field, but this could be another
        # stripped down kind of state for testing.
        test_sys = LennardJonesPair()

        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        reporter = WepyHDF5Reporter(
            file_path=h5_path,
            topology=test_sys.json_top,
            **RESAMPLER_REPORTER_ARGS,
            # test output of both an array and a scalar
            save_fields=("positions", "time"),
        )
        reporter.init(**LJ_OPENMM_SIM_COMPONENTS)

        reporter.report(
            **{
                "cycle_idx" : 0,
                "new_walkers" : [
                    Walker(
                        OpenMMState.from_dwim(
                            positions=np.array([
                                [0., 0., 0.,],
                                [1., 1., 1.,],
                            ]) * openmm.unit.nanometer,
                            time=(1.0 * openmm.unit.picosecond),
                        ),
                        0.5,
                    ),
                    Walker(
                        OpenMMState.from_dwim(
                            positions=np.array([
                                [1., 1., 1.,],
                                [0., 0., 0.,],
                            ]) * openmm.unit.nanometer,
                            time=(1.0 * openmm.unit.picosecond),
                        ),
                        0.5,
                    ),
                ],
                "n_segment_steps" : 100,
                # Instead of cloning and merging we just swap their
                # positions to add a little more reality to the test
                "resampled_walkers" : [
                    Walker(
                        OpenMMState.from_dwim(
                            positions=np.array([
                                [1., 1., 1.,],
                                [0., 0., 0.,],
                            ]) * openmm.unit.nanometer,
                            time=(1.0 * openmm.unit.picosecond),
                        ),
                        0.5,
                    ),
                    Walker(
                        OpenMMState.from_dwim(
                            positions=np.array([
                                [0., 0., 0.,],
                                [1., 1., 1.,],
                            ]) * openmm.unit.nanometer,
                            time=(1.0 * openmm.unit.picosecond),
                        ),
                        0.5,
                    ),
                ],
                "resampling_data" : [
                    {
                        "decision_id" : np.array([0]),
                        "target_idxs" : np.array([[1]]),
                        "step_idx" : np.array([0]),
                        "walker_idx" : np.array([0]),
                    },
                    {
                        "decision_id" : np.array([0]),
                        "target_idxs" : np.array([[0]]),
                        "step_idx" : np.array([0]),
                        "walker_idx" : np.array([1]),
                    },
                ],
            },
            **CYCLE_REPORT_DICT_COMMON,
            **CYCLE_REPORT_DICT_EMPTY_OPTIONALS,
        )

        with reporter.wepy_h5 as wepy_h5:

            assert len(wepy_h5.h5['runs/0/trajectories']) == 2

            assert "0" in wepy_h5.h5['runs/0/trajectories']
            assert "1" in wepy_h5.h5['runs/0/trajectories']

            assert "weights" in wepy_h5.h5['runs/0/trajectories/0']
            assert "positions" in wepy_h5.h5['runs/0/trajectories/0']
            assert "time" in wepy_h5.h5['runs/0/trajectories/0']

            assert wepy_h5.h5['runs/0/trajectories/0/positions'].shape == (1,2,3)
            assert wepy_h5.h5['runs/0/trajectories/0/time'].shape == (1,1)
