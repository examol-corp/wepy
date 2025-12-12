from pathlib import Path
import json
from typing import Callable
import pytest
import numpy as np
from unittest.mock import patch, PropertyMock


import h5py
from wepy.hdf5 import (
    numpy_dtype_to_json,
    dtype_json_to_numpy,
    WepyHDF5,
    _iter_field_paths,
)
from wepy.walker import Walker, WalkerStateBox

from wepy_tools.systems.lennard_jones import LennardJonesPair

@pytest.fixture(scope="session")
def wepy_h5_factory() -> Callable[[Path], WepyHDF5]:

    test_sys = LennardJonesPair()

    def _factory(path: Path) -> Path:

        # create the file
        WepyHDF5(
            path,
            mode="x",
            topology=test_sys.json_top,
        )

        return path

    return _factory

_INIT_WALKERS = [
                    Walker(
                        WalkerStateBox(
                            positions=np.array([
                                [1., 1., 1.],
                                [2., 2., 2.],
                            ]),
                            box_vectors=np.array([
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.],
                            ]),
                            kinetic_energy=3.455,
                        ),
                        0.1,
                    ),
                ]

# @pytest.fixture(scope="session")
# def wepy_h5_file_ro(tmpdir) -> WepyHDF5:
#     """Read-only WepyHDF5"""

#     return gen_wepy_h5(Path(tmpdir) / "main.wepy.h5")

# @pytest.fixture(scope="function")
# def wepy_h5_file_rw(tmpdir):
#     """Read-write WepyHDF5"""
#     return gen_wepy_h5(
#         Path(tmpdir) / "main.wepy.h5",
#     )


# # TODO: this will be easier once we have a fixture for a full WepyHDF5
# def test__iter_field_paths(wepy_h5_file_ro):
#     pass

def test_numpy_dtype_to_json():
    assert json.loads(
        numpy_dtype_to_json(np.dtype(np.int32))
    ) == {
        "kind" : "simple",
        "str" : "<i4",
    }
    assert json.loads(
        numpy_dtype_to_json(np.dtype(np.int64))
    ) == {
        "kind" : "simple",
        "str" : "<i8",
    }

    assert json.loads(
        numpy_dtype_to_json(np.dtype(np.float32))
    ) == {
        "kind" : "simple",
        "str" : "<f4",
    }
    
    assert json.loads(
        numpy_dtype_to_json(np.dtype(np.float64))
    ) == {
        "kind" : "simple",
        "str" : "<f8",
    }

def test_dtype_json_to_numpy():

    assert dtype_json_to_numpy(
        json.dumps(
            {
                "kind" : "simple",
                "str" : "<i4",
            }
        )
    ) == np.dtype(np.int32)

    assert dtype_json_to_numpy(
        json.dumps(
            {
                "kind" : "simple",
                "str" : "<i8",
            }
        )
    ) == np.dtype(np.int64)

    assert dtype_json_to_numpy(
        json.dumps(
            {
                "kind" : "simple",
                "str" : "<f4",
            }
        )
    ) == np.dtype(np.float32)

    assert dtype_json_to_numpy(
        json.dumps(
            {
                "kind" : "simple",
                "str" : "<f8",
            }
        )
    ) == np.dtype(np.float64)
    
class Test_WepyHDF5:

    def test___init__(self, tmp_path_factory):

        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        with pytest.raises(ValueError):
            WepyHDF5(
                h5_path,
                mode="P",
            )
        
        with pytest.raises(ValueError):
            WepyHDF5(
                h5_path,
                mode="x",
            )
        with pytest.raises(ValueError):
            WepyHDF5(
                h5_path,
                mode="w",
            )
        with pytest.raises(ValueError):
            WepyHDF5(
                h5_path,
                mode="w-",
            )

        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        assert wh5.filename.exists()

        with pytest.raises(FileExistsError):
            WepyHDF5(
                h5_path,
                mode="x",
                topology=test_sys.json_top,
            )

        WepyHDF5(
            h5_path,
            mode="r",
        )

        with pytest.raises(ValueError):
            WepyHDF5(
                h5_path,
                mode="r",
                topology=test_sys.json_top,
            )

        # NOTE: tests on data conformance see other tests, we first
        # need to bootstrap other functionality in these unit tests
        # before getting there.

    def test_set_mode(self, tmp_path_factory):

        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"
        
        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        assert wh5.mode == "x"
        wh5.set_mode("r")
        assert wh5.mode == "r"

        # wh5.closed = False
        # with pytest.raises(RuntimeError):
        #     wh5.set_mode("r+")

    def test_open(self, tmp_path_factory):
        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        assert wh5.closed == True
        assert wh5._wepy_mode == "x"


        with pytest.raises(FileExistsError):
            wh5.open()
        with pytest.raises(FileExistsError):
            wh5.open(mode="x")
        with pytest.raises(FileExistsError):
            wh5.open(mode="w-")

        wh5.set_mode('r')
        wh5.open()
        assert wh5.closed == False
        assert wh5._wepy_mode == "r"
        with pytest.raises(IOError):
            wh5.open()

        h5_path = d0 / "main2.wepy.h5"
        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        wh5.open(mode='r+')
        assert wh5.closed == False
        assert wh5._wepy_mode == "r+"


    def test_close(self, tmp_path_factory):
        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")
        h5_path = d0 / "main.wepy.h5"

        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        wh5.open("r")

        assert wh5.closed == False
        wh5.close()
        assert wh5.closed == True


    def test___del__(self, tmp_path_factory):
        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")
        
        h5_path = d0 / "main.wepy.h5"
        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )
        del wh5

        h5_path = d0 / "2.wepy.h5"
        wh5 = WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        wh5.open('r+')
        del wh5

    def test_context_manager(self, tmp_path_factory):
        test_sys = LennardJonesPair()
        d0 = tmp_path_factory.mktemp("0")

        h5_path = d0 / "main.wepy.h5"
        # create
        WepyHDF5(
            h5_path,
            mode="x",
            topology=test_sys.json_top,
        )

        wepy_h5 = WepyHDF5(
            h5_path,
            mode="r",
        )
        with wepy_h5:

            assert wepy_h5.closed == False

        assert wepy_h5.closed == True

    def test__gen_default_init_field_attributes(self):
        test_sys = LennardJonesPair()

        field_feature_shapes, field_feature_dtypes, n_dims, n_coords, main_rep_idxs = WepyHDF5._gen_default_init_field_attributes(
            test_sys.json_top,
            None,
            None,
        )

        assert field_feature_shapes == {
                "time" : (1,),
                "box_vectors" : (3, 3),
                "box_volume" : (1,),
                "kinetic_energy" : (1,),
                "potential_energy" : (1,),
                "positions" : (2, 3),
                "velocities" : (2, 3),
                "forces" : (2, 3),
            }
        
        assert field_feature_dtypes == {
                "time" : float,
                "box_vectors" : float,
                "box_volume" : float,
                "kinetic_energy" : float,
                "potential_energy" : float,
                "positions" : float,
                "velocities" : float,
                "forces" : float,
            }
        assert n_dims == 3
        assert n_coords == 2
        assert np.array_equal(main_rep_idxs, np.array([0,1]))

        field_feature_shapes, _, _, n_coords, main_rep_idxs = WepyHDF5._gen_default_init_field_attributes(
            test_sys.json_top,
            [0,1,2,3],
            None,
        )

        assert n_coords == 4
        assert np.array_equal(main_rep_idxs, np.array([0,1,2,3]))
        assert field_feature_shapes == {
                "time" : (1,),
                "box_vectors" : (3, 3),
                "box_volume" : (1,),
                "kinetic_energy" : (1,),
                "potential_energy" : (1,),
                "positions" : (4, 3),
                "velocities" : (4, 3),
                "forces" : (4, 3),
            }

        field_feature_shapes, _, n_dims, _, _ = WepyHDF5._gen_default_init_field_attributes(
            test_sys.json_top,
            None,
            4,
        )

        assert field_feature_shapes == {
                "time" : (1,),
                "box_vectors" : (3, 3),
                "box_volume" : (1,),
                "kinetic_energy" : (1,),
                "potential_energy" : (1,),
                "positions" : (2, 4),
                "velocities" : (2, 4),
                "forces" : (2, 4),
            }
        assert n_dims == 4

    def test__init_continuations(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")

        h5_path = d0 / "test.h5"

        h5 = h5py.File(h5_path, mode="x")

        h5.create_group("_settings")

        dset = WepyHDF5._init_continuations(
            h5,
        )

        assert "continuations" in h5["_settings"]
        assert dset.shape == (0,2)
        assert dset.dtype == np.int64

    def test__create_init(self, tmp_path_factory):

        test_sys = LennardJonesPair()

        d0 = tmp_path_factory.mktemp("0")

        h5_path = d0 / "test1.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            sparse_fields=None,
        )

        assert "_settings" in h5

        assert "n_dims" in h5["_settings"]
        assert h5["_settings/n_dims"][()] == np.int64(3)
        
        assert "n_atoms" in h5["_settings"]
        assert h5["_settings/n_atoms"][()] == np.int64(2)
        
        assert "main_rep_idxs" in h5["_settings"]
        assert np.array_equal(
            h5["_settings/main_rep_idxs"][:],
            np.array([0,1]),
        )
        
        assert "alt_reps_idxs" in h5["_settings"]
        assert len(h5["_settings/alt_reps_idxs"]) == 0
        
        assert "field_feature_shapes" in h5["_settings"]
        assert "field_feature_dtypes" in h5["_settings"]

        default_fields = {
            "time",
            "box_vectors",
            "box_volume",
            "kinetic_energy",
            "potential_energy",
            "positions",
            "velocities",
            "forces",
        }

        assert set(h5["_settings/field_feature_dtypes"].keys()) == default_fields
        assert set(h5["_settings/field_feature_shapes"].keys()) == default_fields

        _f64 = {
            "kind" : "simple",
            "str" : "<f8",
        }

        assert json.loads(
            h5["_settings/field_feature_dtypes/time"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/box_vectors"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/box_volume"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/kinetic_energy"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/potential_energy"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/positions"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/velocities"][()].decode()
        ) == _f64
        assert json.loads(
            h5["_settings/field_feature_dtypes/forces"][()].decode()
        ) == _f64

        assert np.array_equal(h5["_settings/field_feature_shapes/time"], np.array([1]))
        assert np.array_equal(h5["_settings/field_feature_shapes/box_vectors"], np.array([3, 3]))
        assert np.array_equal(h5["_settings/field_feature_shapes/box_volume"], np.array([1]))
        assert np.array_equal(h5["_settings/field_feature_shapes/kinetic_energy"], np.array([1]))
        assert np.array_equal(h5["_settings/field_feature_shapes/potential_energy"], np.array([1]))
        assert np.array_equal(h5["_settings/field_feature_shapes/positions"], np.array([2, 3]))
        assert np.array_equal(h5["_settings/field_feature_shapes/velocities"], np.array([2, 3]))
        assert np.array_equal(h5["_settings/field_feature_shapes/forces"], np.array([2, 3]))

        assert "sparse_fields" in h5["_settings"]
        assert h5["_settings/sparse_fields"].shape == (0,)
        assert h5py.check_string_dtype(h5["_settings/sparse_fields"].dtype) is not None

        assert "record_fields" in h5["_settings"]
        assert len(h5["_settings/record_fields"]) == 0

        assert "continuations" in h5["_settings"]

        assert "units" in h5
        assert len(h5["units"]) == 0

        assert "topology" in h5
        assert h5["topology"].shape == ()
        # just make sure it can be deserialized
        json.loads(h5["topology"][()].decode())

        assert "runs" in h5
        assert len(h5["runs"]) == 0

        h5_path = d0 / "test2.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            sparse_fields=("velocities",),
        )

        assert h5["_settings/sparse_fields"].shape == (1,)
        assert h5["_settings/sparse_fields"][0].decode() == "velocities"

        h5_path = d0 / "test3.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            field_feature_shapes_overrides={
                "thing" : (1,),
            },
            field_feature_dtypes_overrides={
                "thing" : int,
            },
        )

        assert "field_feature_shapes" in h5["_settings"]
        assert "field_feature_dtypes" in h5["_settings"]

        assert h5["_settings/field_feature_dtypes"].keys() == default_fields | {"thing"}

        h5_path = d0 / "test4.h5"
        h5 = h5py.File(h5_path, mode="x")
        with pytest.raises(ValueError):
            WepyHDF5._create_init(
                h5,
                topology=test_sys.json_top,
                field_feature_shapes_overrides={
                    "thing" : (1,),
                },
                field_feature_dtypes_overrides={},
            )

        h5_path = d0 / "test5.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            sparse_fields=("thing",),
        )

        assert np.isnan(h5["_settings/field_feature_shapes/thing"][()])
        assert h5["_settings/field_feature_dtypes/thing"][()].decode() == "None"

        h5_path = d0 / "test6.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            sparse_fields=("thing",),
            field_feature_shapes_overrides={
                "thing" : (1,),
            },
            field_feature_dtypes_overrides={
                "thing" : np.int32,
            },
        )

        assert h5["_settings/field_feature_shapes/thing"][()] == np.array([1])
        assert dtype_json_to_numpy(h5["_settings/field_feature_dtypes/thing"][()].decode()) == np.dtype(np.int32)

        h5_path = d0 / "test7.h5"
        h5 = h5py.File(h5_path, mode="x")
        WepyHDF5._create_init(
            h5,
            topology=test_sys.json_top,
            units={"positions" : "nanometer"},
        )

        assert "positions" in h5["units"]
        assert len(h5["units"]) == 1
        assert h5["units/positions"][()].decode() == "nanometer"

    def test_num_runs(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert wepy_h5.num_runs == 0

            wepy_h5.h5["runs"].create_group("0")
            assert wepy_h5.num_runs == 1


    def test_next_run_idx(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert wepy_h5.next_run_idx() == 0
            wepy_h5.h5["runs"].create_group("0")
            assert wepy_h5.next_run_idx() == 1

    # TODO: see new_run test for the same effect
    # def test__add_init_walkers(self, wepy_h5_factory, tmpdir):
    #     pass

    # def test__add_run_init(self, wepy_h5_factory, tmpdir):
    #     pass

    def test_new_run(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=[
                    Walker(
                        WalkerStateBox(
                            positions=np.array([
                                [1., 1., 1.],
                                [2., 2., 2.],
                            ]),
                            box_vectors=np.array([
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.],
                            ]),
                            kinetic_energy=3.455,
                        ),
                        0.1,
                    ),
                ],
            )

            assert "0" in wepy_h5.h5["runs"]
            assert "init_walkers" in run_grp

            assert len(run_grp["init_walkers"]) == 1
            assert "0" in run_grp["init_walkers"]
            assert set(run_grp["init_walkers/0"].keys()) == {
                "weights",
                "box_vectors",
                "kinetic_energy",
                "positions",
            }

            assert run_grp["init_walkers/0/weights"].shape == (1,1)
            assert run_grp["init_walkers/0/box_vectors"].shape == (1,3,3)
            assert run_grp["init_walkers/0/positions"].shape == (1,2,3)

            # TOREV: this should probably be (1,1) shape, but waiting
            # to see how things shake out later
            assert run_grp["init_walkers/0/kinetic_energy"].shape == (1,)
            
            assert "trajectories" in run_grp

            assert "run_idx" in run_grp.attrs
            assert run_grp.attrs["run_idx"] == 0

            run_grp = wepy_h5.new_run(
                init_walkers=[
                    Walker(
                        WalkerStateBox(),
                        0.1,
                    ),
                ],
                continue_run=0,
                # extra attrs
                foo="hello",
            )

            assert len(wepy_h5.h5["runs"]) == 2
            assert "1" in wepy_h5.h5["runs"]

            assert np.array_equal(
                wepy_h5.h5["_settings/continuations"][:],
                np.array([
                    [1, 0]
                ])
            )

            assert len(run_grp["init_walkers/0"].keys()) == 1
            assert "weights" in run_grp["init_walkers/0"]

            assert run_grp.attrs["run_idx"] == 1
            assert "foo" in run_grp.attrs
            assert run_grp.attrs["foo"] == "hello"

            with pytest.raises(ValueError):
                wepy_h5.new_run(
                    init_walkers=[
                        Walker(
                            WalkerStateBox(),
                            0.1,
                        ),
                    ],
                    run_idx=1,
                )

            with pytest.raises(ValueError):
                wepy_h5.new_run(
                    init_walkers=[],
                )


    # TODO: I know these are working from the WepyHDF5Reporter tests,
    # but these should be tested individually as time permits
    
    # def test_init_run_fields_resampling(self, wepy_h5_factory, tmpdir):
    #     pass

    # def test_init_run_fields_resampling_decision(self):
    #     pass

    # def test_init_run_fields_resampler(self):
    #     pass

    # def test_init_record_fields(self):
    #     pass

    # def test_init_run_fields_warping(self):
    #     pass

    # def test_init_run_fields_progres(self):
    #     pass

    # def test_init_run_fields_bc(self):
    #     pass

    def test_add_traj(self, wepy_h5_factory, tmpdir, monkeypatch):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            traj_grp = wepy_h5.add_traj(
                0,
                data={
                    "positions" : np.array([[
                        [2., 2., 2.,],
                        [1., 1., 1.,],
                    ]]),
                    "box_vectors" : np.array([[
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.],
                            ]]),
                    "kinetic_energy" : np.array([
                        [4.87],
                    ]),
                },
                weights=np.array([[0.2]]),
                metadata={"foo" : "hello"},
            )

            assert "0" in wepy_h5.h5["runs/0/trajectories"]
            assert set(traj_grp.attrs.keys()) == {"run_idx", "traj_idx", "foo"}
            assert traj_grp.attrs["run_idx"] == 0
            assert traj_grp.attrs["traj_idx"] == 0
            assert traj_grp.attrs["foo"] == "hello"

            assert set(traj_grp.keys()) == {
                "weights",
                "positions",
                "box_vectors",
                "kinetic_energy",
            }

            assert traj_grp["weights"].shape == (1,1)
            assert np.array_equal(
                traj_grp["weights"][:],
                np.array([[0.2]]),
            )
            assert traj_grp["positions"].shape == (1,2,3)
            assert traj_grp["box_vectors"].shape == (1,3,3)
            assert traj_grp["kinetic_energy"].shape == (1,1)

            # TOREV: this is an old requirement and should be
            # reviewed. Should use fields defined earlier in
            # initialization. But currently just testing existing
            # behavior

            # must have positions
            with pytest.raises(ValueError):

                wepy_h5.add_traj(
                    0,
                    data={
                        "box_vectors" : np.array([[
                                    [1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                ]]),
                        "kinetic_energy" : np.array([
                            [4.87],
                        ]),
                    },
                )

            # default weights
            traj_grp = wepy_h5.add_traj(
                0,
                data={
                    "positions" : np.array([[
                        [2., 2., 2.,],
                        [1., 1., 1.,],
                    ]]),
                    "box_vectors" : np.array([[
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.],
                            ]]),
                    "kinetic_energy" : np.array([
                        [4.87],
                    ]),
                },
                weights=None,
            )

            assert "1" in wepy_h5.h5["runs/0/trajectories"]
            assert traj_grp.attrs["run_idx"] == 0
            assert traj_grp.attrs["traj_idx"] == 1

            assert "weights" in traj_grp
            assert traj_grp["weights"].shape == (1,1)
            assert np.array_equal(
                traj_grp["weights"][:],
                np.array([[1.]]),
            )

            # TOREV: this can lead to data inconsistency

            # local sparse_idxs
            traj_grp = wepy_h5.add_traj(
                0,
                data={
                    "positions" : np.array([
                        [
                            [2., 2., 2.,],
                            [1., 1., 1.,],
                        ],
                        [
                            [2., 2., 2.,],
                            [1., 1., 1.,],
                        ],
                    ]),
                    "kinetic_energy" : np.array([
                        [4.87],
                    ]),
                },
                sparse_idxs={"kinetic_energy" : [1,]},
            )
            assert "2" in wepy_h5.h5["runs/0/trajectories"]
            assert traj_grp.attrs["run_idx"] == 0
            assert traj_grp.attrs["traj_idx"] == 2

            # NOTE: no box_vectors
            assert set(traj_grp.keys()) == {
                "weights",
                "positions",
                "kinetic_energy",
            }

            assert traj_grp["positions"].shape == (2, 2, 3)

            assert set(traj_grp["kinetic_energy"].keys()) == {"_sparse_idxs", "data"}
            assert traj_grp["kinetic_energy/_sparse_idxs"].shape == (1,)
            assert np.array_equal(
                traj_grp["kinetic_energy/_sparse_idxs"][:],
                np.array([1]),
            )

            assert np.array_equal(
                traj_grp["kinetic_energy/data"][:],
                np.array([[4.87]]),
            )

            # unitialized sparse fields declared in initialization of
            # file

            # HACK: patch in the sparse_fields instead of doing it from scratch
            with patch("wepy.hdf5.WepyHDF5.sparse_fields", new_callable=PropertyMock) as sparse_fields_mock:
                sparse_fields_mock.return_value = np.array(["box_vectors"])

                traj_grp = wepy_h5.add_traj(
                    0,
                    data={
                        "positions" : np.array([
                            [
                                [2., 2., 2.,],
                                [1., 1., 1.,],
                            ],
                            [
                                [2., 2., 2.,],
                                [1., 1., 1.,],
                            ],
                        ]),
                    },
                )
            assert "3" in wepy_h5.h5["runs/0/trajectories"]
            assert traj_grp.attrs["run_idx"] == 0
            assert traj_grp.attrs["traj_idx"] == 3

            assert "box_vectors" in traj_grp
            assert set(traj_grp["box_vectors"].keys()) == {"_sparse_idxs", "data"}
            assert traj_grp["box_vectors/_sparse_idxs"].shape == (0,)
            assert traj_grp["box_vectors/data"].shape == (0,0,0)


    def test_extend_traj(self, wepy_h5_factory, tmpdir):
        pass
# class Test_WepyHDF5_DataConformance:
#     pass
