# Standard Library
from pathlib import Path

# Third Party Library
import pytest

# First Party Library
from wepy.reporter.file import (
    FileReporterABC,
    FileReporterError,
    ProgressiveFileReporterABC,
)
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.runners.mock import MockRunner
from wepy.work_mapper.serial import SerialMapper


class Test_FileReporterABC:

    def test__validate_mode(self):

        assert FileReporterABC._validate_mode("x")
        assert FileReporterABC._validate_mode("w")
        assert FileReporterABC._validate_mode("w-")
        assert FileReporterABC._validate_mode("r")
        assert FileReporterABC._validate_mode("r+")

    def test___init__(self):

        assert FileReporterABC(
            [Path("somewhere/else.txt")],
        ).modes == ["x"]

        FileReporterABC(
            [Path("somewhere/else.txt")],
            ["x"],
        )

        with pytest.raises(FileReporterError):

            FileReporterABC(
                [Path("somewhere/else.txt")],
                ["B"],
            )

    def test_set_mode(self):

        reporter = FileReporterABC(
            [Path("thing.txt")],
            ["r"],
        )
        assert reporter.modes[0] == "r"
        reporter.set_mode(0, "w")
        assert reporter.modes[0] == "w"


class Test_ProgressiveFileReporterABC:

    def test_init(self, tmp_path_factory):

        sim_components = {
            "init_walkers": [],
            "runner": MockRunner(),
            "resampler": NoResampler(),
            "boundary_conditions": None,
            "work_mapper": SerialMapper(),
            "reporters": [],
            "continue_run": None,
        }

        d0 = tmp_path_factory.mktemp("0")

        reporter = ProgressiveFileReporterABC(
            [d0 / "a.txt"],
            ["x"],
        )

        assert reporter.modes[0] == "x"
        reporter.init(**sim_components)
        assert reporter.modes[0] == "w"

        reporter = ProgressiveFileReporterABC(
            [d0 / "a.txt"],
            ["w-"],
        )

        assert reporter.modes[0] == "w-"
        reporter.init(**sim_components)
        assert reporter.modes[0] == "w"

        # for a file that already exists this is an error
        (d0 / "a.txt").write_text("Hello")

        reporter = ProgressiveFileReporterABC(
            [d0 / "a.txt"],
            ["w-"],
        )

        with pytest.raises(FileExistsError):
            reporter.init(**sim_components)

        # Write mode doesn't change the mode on init
        d1 = tmp_path_factory.mktemp("1")

        reporter = ProgressiveFileReporterABC(
            [d1 / "a.txt"],
            ["w"],
        )

        assert reporter.modes[0] == "w"
        reporter.init(**sim_components)
        assert reporter.modes[0] == "w"
