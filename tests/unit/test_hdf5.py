from pathlib import Path
import pytest

import h5py
from wepy.hdf5 import (
    WepyHDF5,
    _iter_field_paths,
)

from wepy_tools.systems.lennard_jones import LennardJonesPair


# def gen_wepy_h5(path: Path, mode: str) -> WepyHDF5:
#     pass

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

        wh5.closed = False
        with pytest.raises(RuntimeError):
            wh5.set_mode("r+")

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


class Test_WepyHDF5_DataConformance:
    pass
