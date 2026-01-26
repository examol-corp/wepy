# Standard Library
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Third Party Library
import numpy as np
import pytest

# First Party Library
from wepy.hdf5 import WepyHDF5
from wepy.resampling.decisions.no_decision import NoDecision
from wepy.resampling.resamplers.noresampler import (
    NoResampler,
    NoResamplerResamplingRecord,
)
from wepy.typing import IdxArray
from wepy.walker import Walker, WalkerStateBox
from wepy_tools.systems.lennard_jones import LennardJonesPair


def reflink_or_copy(src: Path, dst: Path) -> None:
    """Create a copy-on-write reflink if supported.
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


@pytest.fixture(scope="session")
def wepy_h5_factory() -> Callable[[Path], Path]:

    test_sys = LennardJonesPair()

    def _factory(
        path: Path,
        sparse_fields: tuple[str, ...] | None = None,
        alt_reps: dict[str, IdxArray] | None = None,
        main_rep_idxs: IdxArray | None = None,
    ) -> Path:

        # create the file
        WepyHDF5(
            path,
            mode="x",
            topology=test_sys.json_top,
            sparse_fields=sparse_fields,
            alt_reps=alt_reps,
            main_rep_idxs=main_rep_idxs,
        )

        return path

    return _factory


_INIT_WALKERS = [
    Walker(
        WalkerStateBox(
            positions=np.array(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
            box_vectors=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            kinetic_energy=3.455,
        ),
        0.1,
    ),
]


@pytest.fixture(scope="session")
def _wepy_h5_run_init(wepy_h5_factory, tmp_path_factory) -> Path:
    """Data generation fixture, should not be used by individual tests
    as it is slow. Instead use the fixture that makes a copy for
    read/write in each test function.

    """

    d = tmp_path_factory.mktemp("wepy_h5_run_init")

    # initialize a file
    path = wepy_h5_factory(
        d / "main.wepy.h5",
        sparse_fields={"velocities"},
    )

    # initialize the file
    with WepyHDF5(path, mode="r+") as wepy_h5:

        run_grp = wepy_h5.new_run(init_walkers=_INIT_WALKERS)

        wepy_h5.init_run_fields_resampling_decision(
            0,
            NoDecision.enum_dict_by_name(),
        )
        wepy_h5.init_run_fields_resampling(
            0,
            NoResampler.resampling_fields(),
        )
        wepy_h5.init_record_fields(
            "resampling",
            [name for name, _, _ in NoResampler.resampling_fields()],
        )

        # UGLY: synthetic examples of warping without a real class to use

        # warping as it has hardcoded behavior that should be tested
        warp_fields = [
            ("walker_idx", (1,), int),
            ("target_idx", (1,), int),
            ("weight", (1,), float),
        ]
        wepy_h5.init_run_fields_warping(
            0,
            warp_fields,
        )
        wepy_h5.init_record_fields(
            "warping",
            [name for name, _, _ in warp_fields],
        )

        # minimal progress fields for testing continual records
        progress_fields = [
            # single number for a cycle
            ("ensemble_average", (1,), float),
            # per-walker data
            ("walker_distances", Ellipsis, float),
        ]
        wepy_h5.init_run_fields_progress(
            0,
            progress_fields,
        )
        wepy_h5.init_record_fields(
            "progress",
            [name for name, _, _ in progress_fields],
        )

        # TODO: more fields for resampler records and BC
        # records. These are always optional and strictly accessory so
        # holding off on writing more test cases on these.

    return path


@pytest.fixture(scope="function")
def wepy_h5_run_init(_wepy_h5_run_init, tmpdir) -> Path:

    path = tmpdir / "main.wepy.h5"

    reflink_or_copy(_wepy_h5_run_init, path)

    return path


@pytest.fixture(scope="session")
def _wepy_h5_traj_init(_wepy_h5_run_init, tmp_path_factory) -> Path:
    """Data generation fixture, should not be used by individual tests
    as it is slow. Instead use the fixture that makes a copy for
    read/write in each test function.

    This generates an HDF5 with everything in the run init fixture as
    well as trajectories in the run with a single cycle's worth of
    data.

    """

    # make a copy of the run init H5 file and then mutate to add stuff
    d = tmp_path_factory.mktemp("wepy_h5_traj_init")

    path = d / "main.wepy.h5"
    shutil.copy(
        _wepy_h5_run_init,
        path,
    )

    # Add the new data
    with WepyHDF5(path, mode="r+") as wepy_h5:

        traj0_grp = wepy_h5.add_traj(
            0,
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
            },
            weights=np.array([[0.2]]),
            metadata={"foo": "hello"},
        )

        traj1_grp = wepy_h5.add_traj(
            0,
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
            },
            weights=np.array([[0.2]]),
            metadata={"foo": "hello"},
        )

        wepy_h5.extend_cycle_resampling_records(
            0,
            0,
            [
                # NOTE: that this function requires mappings, and
                # these record types double as mappings via the mixin,
                # so we just use them
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[0],
                    walker_idx=0,
                    step_idx=0,
                ),
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[1],
                    walker_idx=1,
                    step_idx=0,
                ),
            ],
        )

        wepy_h5.extend_cycle_progress_records(
            0,
            0,
            [
                # only a single record for the cycle
                {
                    "ensemble_average": 1.2,
                    "walker_distances": [
                        1.0,
                        1.0,
                    ],
                },
            ],
        )

        # TODO: the other record groups

    return path


@pytest.fixture(scope="function")
def wepy_h5_traj_init(_wepy_h5_traj_init, tmpdir) -> Path:

    path = tmpdir / "main.wepy.h5"

    reflink_or_copy(_wepy_h5_traj_init, path)

    return path


@pytest.fixture(scope="session")
def _wepy_h5_full_init(_wepy_h5_traj_init, tmp_path_factory) -> Path:
    """Data generation fixture, should not be used by individual tests
    as it is slow. Instead use the fixture that makes a copy for
    read/write in each test function.

    This generates an HDF5 with everything in the traj init fixture as
    well as extending those trajectories for a few cycles and adding
    another 1 run. This run is a continuation of the first.

    For each trajectory it also includes sparse data and alternate reps.

    This should be sufficient for testing of all HDF5 related methods
    and analysis.

    """

    # make a copy of the run init H5 file and then mutate to add stuff
    d = tmp_path_factory.mktemp("wepy_h5_full")

    path = d / "main.wepy.h5"
    shutil.copy(
        _wepy_h5_traj_init,
        path,
    )

    # Add the new data
    with WepyHDF5(path, mode="r+") as wepy_h5:

        # extend run 0
        wepy_h5.extend_traj(
            0,
            0,
            weights=np.array([[0.2]]),
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
                "velocities": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
            },
        )

        wepy_h5.extend_traj(
            0,
            1,
            weights=np.array([[0.2]]),
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
                "velocities": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
            },
        )

        wepy_h5.extend_cycle_resampling_records(
            0,
            1,
            [
                # NOTE: that this function requires mappings, and
                # these record types double as mappings via the mixin,
                # so we just use them
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[0],
                    walker_idx=0,
                    step_idx=0,
                ),
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[1],
                    walker_idx=1,
                    step_idx=0,
                ),
            ],
        )

        wepy_h5.extend_cycle_progress_records(
            0,
            1,
            [
                # only a single record for the cycle
                {
                    "ensemble_average": 1.2,
                    "walker_distances": [
                        1.0,
                        1.0,
                    ],
                },
            ],
        )

        # run 1, a continuation of run 0
        wepy_h5.new_run(
            init_walkers=_INIT_WALKERS,
            continue_run=0,
        )

        wepy_h5.init_run_fields_resampling_decision(
            1,
            NoDecision.enum_dict_by_name(),
        )
        wepy_h5.init_run_fields_resampling(
            1,
            NoResampler.resampling_fields(),
        )

        # UGLY: synthetic examples of warping without a real class to use

        # warping as it has hardcoded behavior that should be tested
        warp_fields = [
            ("walker_idx", (1,), int),
            ("target_idx", (1,), int),
            ("weight", (1,), float),
        ]
        wepy_h5.init_run_fields_warping(
            1,
            warp_fields,
        )

        # minimal progress fields for testing continual records
        progress_fields = [
            # single number for a cycle
            ("ensemble_average", (1,), float),
            # per-walker data
            ("walker_distances", Ellipsis, float),
        ]
        wepy_h5.init_run_fields_progress(
            1,
            progress_fields,
        )

        wepy_h5.add_traj(
            1,
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
            },
            weights=np.array([[0.2]]),
            metadata={"foo": "hello"},
        )

        wepy_h5.add_traj(
            1,
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
            },
            weights=np.array([[0.2]]),
            metadata={"foo": "hello"},
        )

        wepy_h5.extend_cycle_resampling_records(
            1,
            0,
            [
                # NOTE: that this function requires mappings, and
                # these record types double as mappings via the mixin,
                # so we just use them
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[0],
                    walker_idx=0,
                    step_idx=0,
                ),
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[1],
                    walker_idx=1,
                    step_idx=0,
                ),
            ],
        )

        wepy_h5.extend_cycle_progress_records(
            1,
            0,
            [
                # only a single record for the cycle
                {
                    "ensemble_average": 1.2,
                    "walker_distances": [
                        1.0,
                        1.0,
                    ],
                },
            ],
        )

        # extend run 1
        wepy_h5.extend_traj(
            1,
            0,
            weights=np.array([[0.2]]),
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
                "velocities": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
            },
        )

        wepy_h5.extend_traj(
            1,
            1,
            weights=np.array([[0.2]]),
            data={
                "positions": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
                "box_vectors": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                "kinetic_energy": np.array(
                    [
                        [4.87],
                    ]
                ),
                "velocities": np.array(
                    [
                        [
                            [
                                2.0,
                                2.0,
                                2.0,
                            ],
                            [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        ]
                    ]
                ),
            },
        )

        wepy_h5.extend_cycle_resampling_records(
            1,
            1,
            [
                # NOTE: that this function requires mappings, and
                # these record types double as mappings via the mixin,
                # so we just use them
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[0],
                    walker_idx=0,
                    step_idx=0,
                ),
                NoResamplerResamplingRecord(
                    decision_id=0,
                    target_idxs=[1],
                    walker_idx=1,
                    step_idx=0,
                ),
            ],
        )

        wepy_h5.extend_cycle_progress_records(
            1,
            1,
            [
                # only a single record for the cycle
                {
                    "ensemble_average": 1.2,
                    "walker_distances": [
                        1.0,
                        1.0,
                    ],
                },
            ],
        )

    return path


@pytest.fixture(scope="function")
def wepy_h5_full_init(_wepy_h5_full_init, tmpdir) -> Path:

    path = tmpdir / "main.wepy.h5"

    reflink_or_copy(_wepy_h5_full_init, path)

    return path
