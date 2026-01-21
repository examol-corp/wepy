import shutil
import subprocess
import sys

from pathlib import Path

import pytest



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

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="function")
def alanine_dipeptide_revo_wepy_hdf5(tmp_path: Path) -> Path:

    src = DATA_DIR / "alanine_dipeptide_revo.wepy.hdf5"
    dst = tmp_path / "data.wepy.hdf5"

    reflink_or_copy(src, dst)
    return dst
