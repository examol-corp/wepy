from pathlib import Path
import json
import sys
import shutil
import subprocess
from typing import Callable
import pytest
import numpy as np
from unittest.mock import patch, PropertyMock


import h5py
from wepy.typing import IdxArray
from wepy.hdf5 import (
    numpy_dtype_to_json,
    dtype_json_to_numpy,
    WepyHDF5,
    _iter_field_paths,
    WepyHDF5Error,
    WepyHDF5WriteError,
    WepyHDF5ReadError,
)
from wepy.storage.protocol import RunRecord
from wepy.walker import Walker, WalkerStateBox
from wepy.resampling.decisions.no_decision import (
    NoDecision,
)
from wepy.resampling.resamplers.noresampler import NoResampler, NoResamplerResamplingRecord

from wepy_tools.systems.lennard_jones import LennardJonesPair

def reflink_or_copy(src: Path, dst: Path) -> None:
    """
    Create a copy-on-write reflink if supported.
    Fall back to a full copy otherwise.
    """
    try:
        if sys.platform.startswith("linux"):
            subprocess.run(
                ["cp", "--reflink=auto", src, dst],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "darwin":
            subprocess.run(
                ["cp", "-c", src, dst],  # APFS clone
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            raise RuntimeError("No reflink support")
    except Exception:
        shutil.copy2(src, dst)



       


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

    def test_settings_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert wepy_h5.settings_grp == wepy_h5.h5["_settings"]


    def test__add_init_walkers(self, wepy_h5_factory, tmpdir):
        assert False

    def test__add_run_init(self, wepy_h5_factory, tmpdir):
        pass


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

    def test__init_run_records_field(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            record_grp = run_grp.create_group("example")

            a_dset = wepy_h5._init_run_records_field(
                0,
                "example",
                field_name="a",
                field_shape=(1,),
                field_dtype=np.int64,
            )

            assert "a" in record_grp
            assert a_dset.shape == (0,1)
            assert a_dset.dtype == np.int64
            assert a_dset.maxshape == (None, 1)

            b_dset = wepy_h5._init_run_records_field(
                0,
                "example",
                field_name="b",
                field_shape=(3,3),
                field_dtype=np.float64,
            )

            assert "b" in record_grp
            assert b_dset.shape == (0,3,3)
            assert b_dset.dtype == np.float64
            assert b_dset.maxshape == (None, 3,3)

            c_dset = wepy_h5._init_run_records_field(
                0,
                "example",
                field_name="c",
                field_shape=Ellipsis,
                field_dtype=bool,
            )

            assert "c" in record_grp
            assert c_dset.shape == (0,)
            assert h5py.check_vlen_dtype(c_dset.dtype) == bool
            assert c_dset.maxshape == (None,)


    def test__init_run_sporadic_record_grp(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            record_grp = wepy_h5._init_run_sporadic_record_grp(
                0,
                "example",
                [
                    ("a", (1,), np.float64),
                    ("b", (3, 3), np.int32),
                ],
            )

            assert "runs/0/example" in wepy_h5.h5

            assert "_cycle_idxs" in record_grp
            assert record_grp["_cycle_idxs"].shape == (0,)
            assert record_grp["_cycle_idxs"].maxshape == (None,)
            assert record_grp["_cycle_idxs"].dtype == np.int64

            assert "a" in record_grp
            assert "b" in record_grp

    def test__init_run_continual_record_grp(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            record_grp = wepy_h5._init_run_continual_record_grp(
                0,
                "example",
                (
                    ("a", (3, 3,), np.float32),
                    ("b", Ellipsis, np.int32),
                ),
            )

            assert "example" in run_grp

            assert "_cycle_idxs" not in record_grp
            assert "a" in record_grp
            assert "b" in record_grp

            assert record_grp["a"].shape == (0, 3, 3)
            assert record_grp["a"].maxshape == (None, 3, 3)
            assert record_grp["a"].dtype == np.float32

            assert record_grp["b"].shape == (0,)
            assert record_grp["b"].maxshape == (None,)
            assert h5py.check_vlen_dtype(record_grp["b"].dtype) == np.int32
            
    def test__is_sporadic_records(self):

        assert WepyHDF5._is_sporadic_records("resampler")
        assert WepyHDF5._is_sporadic_records("warping")
        assert WepyHDF5._is_sporadic_records("resampling")
        assert WepyHDF5._is_sporadic_records("boundary_conditions")

        # everything else...
        assert not WepyHDF5._is_sporadic_records("example")

    def test_init_run_record_grp(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            # continual
            example_grp = wepy_h5.init_run_record_grp(
                0,
                "example",
                [
                    ("a", (1,), np.float64),
                    ("b", (3, 3), np.int32),
                ],
            )

            assert "runs/0/example" in wepy_h5.h5
            assert "_cycle_idxs" not in example_grp

            # sporadic
            resampling_grp = wepy_h5.init_run_record_grp(
                0,
                "resampling",
                [
                    ("a", (1,), np.float64),
                    ("b", (3, 3), np.int32),
                ],
            )

            assert "runs/0/resampling" in wepy_h5.h5
            assert "_cycle_idxs" in resampling_grp
            
    def test_init_run_fields_resampling(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            resampling_grp = wepy_h5.init_run_fields_resampling(
                0,
                [
                    ("decision_id", (1,), np.uint32),
                    ("target_idxs", Ellipsis, np.uint32),
                ],
            )

            assert "resampling" in wepy_h5.h5["runs/0"]
            assert "_cycle_idxs" in resampling_grp


    # TODO: I know these are working from the WepyHDF5Reporter tests,
    # but these should be tested individually as time permits
    
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



    def test_runs(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:
            assert wepy_h5.runs == wepy_h5.h5["runs"]

    def test_run(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.run(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            assert wepy_h5.run(0) == run_grp

    def test_decision_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            
            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.decision_grp(0)

            decision_grp = wepy_h5.init_run_fields_resampling_decision(
                0,
                NoDecision.enum_dict_by_name(),
            )

            assert wepy_h5.decision_grp(0) == run_grp["decision"]

    def test_init_walkers_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.init_walkers_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            assert wepy_h5.init_walkers_grp(0) == run_grp["init_walkers"]

    def test_records_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(KeyError):
                wepy_h5.records_grp(0, "notarecordgroup")

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.records_grp(0, "resampling")

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            with pytest.raises(KeyError):
                wepy_h5.records_grp(0, "trajectories")

            run_grp.create_group("resampling")
            run_grp.create_group("resampler")
            run_grp.create_group("warping")
            run_grp.create_group("boundary_conditions")
            run_grp.create_group("progress")
            assert wepy_h5.records_grp(0, "resampling") == run_grp["resampling"]
            assert wepy_h5.records_grp(0, "resampler") == run_grp["resampler"]
            assert wepy_h5.records_grp(0, "warping") == run_grp["warping"]
            assert wepy_h5.records_grp(0, "boundary_conditions") == run_grp["boundary_conditions"]
            assert wepy_h5.records_grp(0, "progress") == run_grp["progress"]

    def test_resampling_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.resampling_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            run_grp.create_group("resampling")
            assert wepy_h5.resampling_grp(0) == run_grp["resampling"]

    def test_resampler_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.resampler_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            run_grp.create_group("resampler")
            
            assert wepy_h5.records_grp(0, "resampler") == run_grp["resampler"]
    def test_warping_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.warping_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            run_grp.create_group("warping")
            assert wepy_h5.records_grp(0, "warping") == run_grp["warping"]
    def test_bc_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.bc_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            run_grp.create_group("boundary_conditions")
            assert wepy_h5.records_grp(0, "boundary_conditions") == run_grp["boundary_conditions"]
    def test_progress_grp(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.progress_grp(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            run_grp.create_group("progress")
            assert wepy_h5.records_grp(0, "progress") == run_grp["progress"]

    def test_run_trajs(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.run_trajs(0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            assert wepy_h5.run_trajs(0) == run_grp["trajectories"]

    def test_traj(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.traj(0, 0)

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.traj(0, 0)

            traj_grp = run_grp["trajectories"].create_group("0")

            assert wepy_h5.traj(0, 0) == traj_grp

    def test_traj_field_entity(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.traj_field_entity(0, 0, "something")

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.traj_field_entity(0, 0, "something")

            traj_grp = run_grp["trajectories"].create_group("0")

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.traj_field_entity(0, 0, "something")

            field_grp = traj_grp.create_group("something")
            
            assert wepy_h5.traj_field_entity(0, 0, "something") == field_grp

            field_dset = traj_grp.create_dataset("dset", dtype=np.float32, shape=(0, 0))
            assert wepy_h5.traj_field_entity(0, 0, "dset") == field_dset
            
            
    def test_num_runs(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert wepy_h5.num_runs == 0

            wepy_h5.h5["runs"].create_group("0")
            assert wepy_h5.num_runs == 1

    def test_num_run_trajs(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.num_run_trajs(0)

            run_grp = wepy_h5.h5["runs"].create_group("0")
                
            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.num_run_trajs(0)

            run_grp.create_group("trajectories/0")

            assert wepy_h5.num_run_trajs(0) == 1

            run_grp.create_group("trajectories/1")
            assert wepy_h5.num_run_trajs(0) == 2
            

    def test_next_run_idx(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert wepy_h5.next_run_idx() == 0
            wepy_h5.h5["runs"].create_group("0")
            assert wepy_h5.next_run_idx() == 1

    def test_next_run_traj_idx(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5.next_run_traj_idx(0)
    
            run_grp = wepy_h5.h5["runs"].create_group("0")
            run_grp.create_group("trajectories")

            assert wepy_h5.next_run_traj_idx(0) == 0

            run_grp.create_group("trajectories/0")

            assert wepy_h5.next_run_traj_idx(0) == 1

            run_grp.create_group("trajectories/1")
            assert wepy_h5.next_run_traj_idx(0) == 2

    def test__extend_run_record_data_field(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            resampling_grp = wepy_h5.init_run_fields_resampling(
                0,
                [
                    ("decision_id", (1,), np.uint32),
                    ("target_idxs", Ellipsis, np.uint32),
                    ("step_idx", (1,), np.uint32),
                    ("walker_idx", (1,), np.uint32),
                ],
            )

            assert resampling_grp["decision_id"].shape == (0,1)
            wepy_h5._extend_run_record_data_field(
                0,
                "resampling",
                "decision_id",
                np.array([[0]]),
            )

            assert resampling_grp["decision_id"].shape == (1,1)
            assert np.array_equal(
                resampling_grp["decision_id"][:],
                np.array([
                    [0]
                ]),
            )

            wepy_h5._extend_run_record_data_field(
                0,
                "resampling",
                "decision_id",
                np.array([[0]]),
            )

            assert resampling_grp["decision_id"].shape == (2,1)
            assert np.array_equal(
                resampling_grp["decision_id"][:],
                np.array([
                    [0],
                    [0],
                ]),
            )
            
        
    def test_extend_cycle_run_group_records(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            resampling_grp = wepy_h5.init_run_fields_resampling(
                0,
                [
                    ("decision_id", (1,), np.uint32),
                    ("target_idxs", Ellipsis, np.uint32),
                    ("step_idx", (1,), np.uint32),
                    ("walker_idx", (1,), np.uint32),
                ],
            )

            assert resampling_grp["_cycle_idxs"].shape[0] == 0
            assert resampling_grp["decision_id"].shape[0] == 0
            assert resampling_grp["target_idxs"].shape[0] == 0

            wepy_h5.extend_cycle_run_group_records(
                0,
                "resampling",
                0,
                [
                    {
                        "decision_id" : np.array([[0]]),
                        "target_idxs" : np.array([[[0]]]),
                        "step_idx" : np.array([[0]]),
                        "walker_idx" : np.array([[0]]),
                    },
                    {
                        "decision_id" : np.array([[0]]),
                        "target_idxs" : np.array([[[0]]]),
                        "step_idx" : np.array([[0]]),
                        "walker_idx" : np.array([[1]]),
                    },
                ]
            )

            assert resampling_grp["decision_id"].shape == (2,1)
            assert resampling_grp["target_idxs"].shape == (2,)
            assert resampling_grp["step_idx"].shape == (2,1)
            assert resampling_grp["walker_idx"].shape == (2,1)

    def test_extend_cycle_resampling_records(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            resampling_grp = wepy_h5.init_run_fields_resampling(
                0,
                [
                    ("decision_id", (1,), np.uint32),
                    ("target_idxs", Ellipsis, np.uint32),
                    ("step_idx", (1,), np.uint32),
                    ("walker_idx", (1,), np.uint32),
                ],
            )

            # UGLY,TOREV: This is ugly because the resampler then
            # needs to handle processing the records into these deeply
            # bracketed arrays. However, this is the interface and
            # promised shapes given their interfaces in the
            # e.g. Resampler components and all the downstream tools
            # will rely on this structure so it must stay.
            wepy_h5.extend_cycle_resampling_records(
                0,
                0,
                [
                    {
                        "decision_id" : 0,
                        "target_idxs" : [0],
                        "step_idx" : 0,
                        "walker_idx" : 0,
                    },
                    {
                        "decision_id" : 0,
                        "target_idxs" : [0],
                        "step_idx" : 0,
                        "walker_idx" : 1,
                    },
                ]
            )

            assert resampling_grp["decision_id"].shape == (2,1)
            assert resampling_grp["target_idxs"].shape == (2,)
            assert resampling_grp["step_idx"].shape == (2,1)
            assert resampling_grp["walker_idx"].shape == (2,1)

    def test_sparse_fields(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(
            Path(tmpdir) / "nosparse.wepy.h5",
        )
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert len(wepy_h5.sparse_fields) == 0

        path = wepy_h5_factory(
            Path(tmpdir) / "sparse.wepy.h5",
            sparse_fields=("sparse_thing",),
        )
        with WepyHDF5(path, mode="r+") as wepy_h5:

            assert len(wepy_h5.sparse_fields) == 1
            assert set(wepy_h5.sparse_fields) == {"sparse_thing"}

    def test__init_contiguous_traj_field(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            traj_grp = run_grp.create_group("trajectories/0")

            wepy_h5._init_contiguous_traj_field(
                0,
                0,
                "something",
                (2, 3),
                np.float32,
            )
            assert "something" in traj_grp
            assert traj_grp["something"].maxshape == (None, 2, 3)
            assert traj_grp["something"].shape == (0, 0, 0)

    def test__init_sparse_traj_field(self, wepy_h5_factory, tmpdir):
        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            traj_grp = run_grp.create_group("trajectories/0")

            wepy_h5._init_sparse_traj_field(
                0,
                0,
                "something",
                (2, 3),
                np.float32,
            )
            assert "something" in traj_grp
            assert "data" in traj_grp["something"]
            assert "_sparse_idxs" in traj_grp["something"]

            assert traj_grp["something/data"].maxshape == (None, 2, 3)
            assert traj_grp["something/data"].shape == (0, 0, 0)

            assert traj_grp["something/_sparse_idxs"].maxshape == (None,)
            assert traj_grp["something/_sparse_idxs"].shape == (0,)
            assert traj_grp["something/_sparse_idxs"].dtype == int

    def test__init_traj_field(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(
            Path(tmpdir) / "0.wepy.h5",
            sparse_fields=("sparse_thing",),
        )
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            traj_grp = run_grp.create_group("trajectories/0")

            wepy_h5._init_traj_field(
                0, 0,
                "thing",
                (2, 2),
                np.float32,
            )

            assert "thing" in traj_grp
            assert traj_grp["thing"].maxshape == (None, 2, 2)
            assert traj_grp["thing"].shape == (0, 0, 0)

            wepy_h5._init_traj_field(
                0, 0,
                "sparse_thing",
                (2, 2),
                np.float32,
            )
            assert "sparse_thing" in traj_grp
            assert "data" in traj_grp["sparse_thing"]
            assert "_sparse_idxs" in traj_grp["sparse_thing"]

            assert traj_grp["sparse_thing/data"].maxshape == (None, 2, 2)
            assert traj_grp["sparse_thing/data"].shape == (0, 0, 0)

            assert traj_grp["sparse_thing/_sparse_idxs"].maxshape == (None,)
            assert traj_grp["sparse_thing/_sparse_idxs"].shape == (0,)
            assert traj_grp["sparse_thing/_sparse_idxs"].dtype == int

    def test__init_traj_fields(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(
            Path(tmpdir) / "0.wepy.h5",
            sparse_fields=("sparse_thing",),
        )
        with WepyHDF5(path, mode="r+") as wepy_h5:

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )
            traj_grp = run_grp.create_group("trajectories/0")

            wepy_h5._init_traj_fields(
                0, 0,
                field_paths=["thing", "sparse_thing"],
                field_feature_shapes=[(2, 2), (2, 2)],
                field_feature_dtypes=[np.float32, np.float32],
            )

            assert "thing" in traj_grp
            assert traj_grp["thing"].maxshape == (None, 2, 2)
            assert traj_grp["thing"].shape == (0, 0, 0)

            assert "sparse_thing" in traj_grp
            assert "data" in traj_grp["sparse_thing"]
            assert "_sparse_idxs" in traj_grp["sparse_thing"]

            assert traj_grp["sparse_thing/data"].maxshape == (None, 2, 2)
            assert traj_grp["sparse_thing/data"].shape == (0, 0, 0)

            assert traj_grp["sparse_thing/_sparse_idxs"].maxshape == (None,)
            assert traj_grp["sparse_thing/_sparse_idxs"].shape == (0,)
            assert traj_grp["sparse_thing/_sparse_idxs"].dtype == int

    def test__add_traj_field_data(self, wepy_h5_factory, tmpdir):

        path = wepy_h5_factory(Path(tmpdir) / "0.wepy.h5")
        with WepyHDF5(path, mode="r+") as wepy_h5:

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5._add_traj_field_data(
                    0, 0, "something",
                    np.array([1, 2, 3]),
                )

            run_grp = wepy_h5.new_run(
                init_walkers=_INIT_WALKERS
            )

            with pytest.raises(WepyHDF5ReadError):
                wepy_h5._add_traj_field_data(
                    0, 0, "something",
                    np.array([1, 2, 3]),
                )

            traj_grp = run_grp["trajectories"].create_group("0")

            wepy_h5._add_traj_field_data(
                0, 0, "something",
                np.array([1, 2, 3]),
            )

            assert "something" in traj_grp
            assert list(traj_grp["something"][:]) == [1, 2, 3]

            # overwrite should work if it is the same shape
            wepy_h5._add_traj_field_data(
                0, 0, "something",
                np.array([3, 2, 1]),
            )
            assert list(traj_grp["something"][:]) == [3, 2, 1]

            # but not for different sizes
            with pytest.raises(TypeError):
                wepy_h5._add_traj_field_data(
                    0, 0, "something",
                    np.array([3, 2, 1, 0]),
                )

            # sparse field
            wepy_h5._add_traj_field_data(
                0, 0, "sparse_thing",
                np.array([3, 2, 1]),
                sparse_idxs=np.array([5, 10, 15]),
            )

            assert "sparse_thing" in traj_grp
            assert "data" in traj_grp["sparse_thing"]
            assert "_sparse_idxs" in traj_grp["sparse_thing"]

            assert list(traj_grp["sparse_thing/data"][:]) == [3, 2, 1]
            assert list(traj_grp["sparse_thing/_sparse_idxs"][:]) == [5, 10, 15]

    def test_add_traj(self, wepy_h5_factory, tmpdir):
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

            assert traj_grp["weights"].shape == (1,1)
            assert traj_grp["positions"].shape == (1,2,3)
            assert traj_grp["box_vectors"].shape == (1,3,3)
            assert traj_grp["kinetic_energy"].shape == (1,1)
            
            # then extend this trajectory with all values
            wepy_h5.extend_traj(
                0,
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
            )

            assert traj_grp["weights"].shape == (2,1)
            assert traj_grp["positions"].shape == (2,2,3)
            assert traj_grp["box_vectors"].shape == (2,3,3)
            assert traj_grp["kinetic_energy"].shape == (2,1)

        # sparse fields

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

    def test_record_fields(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:
            
            assert "resampling" in wepy_h5.record_fields
            assert wepy_h5.record_fields["resampling"] == [
                "decision_id",
                "target_idxs",
                "step_idx",
                "walker_idx",
            ]
        

            
    def test__convert_record_field_to_table_column(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5._convert_record_field_to_table_column(
                0, "resampling", "walker_idx"
            ) == [0, 1]

            assert wepy_h5._convert_record_field_to_table_column(
                0, "resampling", "step_idx"
            ) == [0, 0]

            assert wepy_h5._convert_record_field_to_table_column(
                0, "resampling", "target_idxs"
            ) == [(0,), (1,)]
    
    def test__convert_record_fields_to_table_columns(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5._convert_record_fields_to_table_columns(
                0, "resampling"
            ) == {
                "walker_idx" : [0, 1],
                "step_idx" : [0, 0],
                "target_idxs" : [(0,), (1,)],
                "decision_id" : [0, 0],
            }

            progress_cols = wepy_h5._convert_record_fields_to_table_columns(
                0, "progress"
            )
            assert len(progress_cols) == 2
            assert set(progress_cols) == {"ensemble_average", "walker_distances"}
            assert len(progress_cols["ensemble_average"]) == 1
            assert progress_cols["ensemble_average"][0] == 1.2
            assert len(progress_cols["walker_distances"]) == 1
            assert list(progress_cols["walker_distances"][0]) == [1.,1.,]
            
    def test__table_to_run_records(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5._table_to_run_records(
                "resampling",
                {
                    "cycle_idx": [0, 0, 1, 1],
                    "walker_idx" : [0, 1, 0, 1],
                    "step_idx": [0, 0, 0, 0],
                    "target_idxs" : [(0,), (1,), (0,), (1,)],
                    "decision_id": [0, 0, 0, 0],
                },
            ) == [
                # cycle 0
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
                # cycle 1
                RunRecord(
                    cycle_idx=1,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=1,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
            ]
        
    
    def test__run_records_sporadic(self, wepy_h5_traj_init):
        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5._run_records_sporadic(
                [0],
                "resampling",
            ) == [
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
            ]


    def test__run_records_continual(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            recs = wepy_h5._run_records_continual(
                [0],
                "progress",
            )

            assert len(recs) == 1

            assert recs[0].cycle_idx == 0

            assert set(recs[0].record.keys()) == {"ensemble_average", "walker_distances"}
            assert recs[0].record["ensemble_average"] == 1.2
            assert list(recs[0].record["walker_distances"]) == [1., 1.]


    def test_run_contig_records(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            # Sporadic example
            assert wepy_h5.run_contig_records(
                [0],
                "resampling",
            ) == [
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
            ]

            # continual
            progress_recs = wepy_h5.run_contig_records(
                [0],
                "progress",
            )

            assert len(progress_recs) == 1
            assert progress_recs[0].cycle_idx == 0

            # TODO: for multiple runs

    def test_run_records(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5.run_records(0, "resampling") == [
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
            ]

    def test_run_records_dataframe(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            df = wepy_h5.run_records_dataframe(0, "resampling")

            assert set(df.columns) == {"cycle_idx", "walker_idx", "step_idx", "target_idxs", "decision_id"}

    def test_resampling_records(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            assert wepy_h5.resampling_records([0]) == [
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 0,
                        "step_idx" : 0,
                        "target_idxs" : (0,),
                        "decision_id" : 0,
                    },
                ),
                RunRecord(
                    cycle_idx=0,
                    record={
                        "walker_idx" : 1,
                        "step_idx" : 0,
                        "target_idxs" : (1,),
                        "decision_id" : 0,
                    },
                ),
            ]

    
    # TODO
    # def test_resampling_records_dataframe(self, wepy_h5_factory, tmpdir):
    #     pass
    #
    # def test_resampler_records(self, wepy_h5_factory, tmpdir):
    #     pass
    #
    # def test_resampler_records_dataframe(self, wepy_h5_factory, tmpdir):
    #     pass
    
    
    def test_is_run_contig(self, wepy_h5_traj_init):
        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:
            assert wepy_h5.is_run_contig([0])

            # TODO: more complex scenarios

    def test_run_contig_resampling_panel(self, wepy_h5_traj_init):

        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            resampling_panel = wepy_h5.run_contig_resampling_panel([0])
            assert resampling_panel == [
                # cycle 0
                [
                    # step 0
                    [
                        # walker 0
                        {
                            "decision_id": 0,
                            "target_idxs": (0,),
                        },
                        # walker 1
                        {
                            "decision_id": 0,
                            "target_idxs": (1,),
                        },
                    ],
                ],
            ]

    def test_run_resampling_panel(self, wepy_h5_traj_init):
        with WepyHDF5(wepy_h5_traj_init, mode='r') as wepy_h5:

            resampling_panel = wepy_h5.run_resampling_panel(0)
            assert resampling_panel == [
                # cycle 0
                [
                    # step 0
                    [
                        # walker 0
                        {
                            "decision_id": 0,
                            "target_idxs": (0,),
                        },
                        # walker 1
                        {
                            "decision_id": 0,
                            "target_idxs": (1,),
                        },
                    ],
                ],
            ]


    # TODO: for observables
    #
    # def test__add_run_field(self, wepy_h5_factory, tmpdir):
    #     assert False
    # def test__add_field(self, wepy_h5_factory, tmpdir):
    #     assert False
    #
    # def test_add_observable(self, wepy_h5_factory, tmpdir):
    #     assert False

# TODO: add some high level acceptance tests for data. Perhaps move to
# another file
#
# class Test_WepyHDF5_DataConformance:
#     pass

