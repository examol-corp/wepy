# Standard Library
import logging
from abc import ABC
from pathlib import Path
from typing import Literal, get_args

# Local Modules
from .base import ReporterError, SimComponentArgs

logger = logging.getLogger(__name__)


class FileReporterError(ReporterError):
    pass


FileMode = Literal["x", "w", "w-", "r", "r+"]


class FileReporterABC(ABC):
    """Abstract reporter that handles specifying file paths for a
    reporter.

    This abstract class doesn't perform any operations that involve
    actually opening file descriptors, but only the validation and
    organization of file paths.

    This provides a uniform API for retrieving file paths from all
    reporters inheriting from it.

    Additionally, FileReporter implements an interface for performing
    a so-called reparametrization of the relevant values associated
    with each file specification (i.e. file path and mode).

    A reparametrization can be performed by calling the
    'reparametrize' method, and can be customized.

    Additionally, there are some customizable class constants than can
    be used in subclasses to control this process including:
    DEFAULT_MODE, SUGGESTED_FILENAME_TEMPLATE,
    DEFAULT_SUGGESTED_EXTENSION, FILE_ORDER, and SUGGESTED_EXTENSIONS.

    The intention is to allow the redefinition of file paths
    dynamically to adapt to changing runtime requirements. Such as
    execution on a separate subtree of a directory hierarchy.

    """

    MODES = tuple(get_args(FileMode))
    """Valid modes accepted for files."""

    DEFAULT_MODE: FileMode = "x"
    """The default mode to set for opening files if none is specified
    (create if doesn't exist, fail if it does.)"""

    SUGGESTED_FILENAME_TEMPLATE: str = "{config}{narration}{reporter_class}.{ext}"
    """Template to use for dynamic reparametrization of file path names.

    The fields in the template are:

    config : indicator of the runtime configuration used

    narration : freeform description of the instance

    reporter_class : the name of the class that produced the
        output. When no specific name is given for a file report generated
        from a reporter this is used to disambiguate, along with the
        extension.

    ext : The file extension, for multiple files produced from one
    reporter this should be sufficient to disambiguate the files.

    The 'config' and 'narration' should be the same across all
    reporters in the same simulation manager, and the 'narration' is
    considered optional.

    """

    DEFAULT_SUGGESTED_EXTENSION: str = "report"
    """The default file extension used for files during dynamic
    reparametrization, if none is specified"""

    FILE_ORDER: tuple[str, ...] = ()
    """Specify an ordering of file paths. Should be customized."""

    SUGGESTED_EXTENSIONS: tuple[str, ...] = ()
    """Suggested extensions for file paths for use with the automatic
    reparametrization feature. Should be customized."""

    @classmethod
    def _validate_mode(cls, mode: FileMode) -> bool:
        """Check if the mode spec is a valid one.

        Parameters
        ----------
        mode : str

        Returns
        -------
        valid : bool

        """
        if mode in cls.MODES:
            return True
        else:
            return False

    def __init__(
        self,
        file_paths: list[Path],
        modes: list[FileMode] | None = None,
    ) -> None:
        """Constructor for FileReporter.

        This constructor allows the specification of either a list of
        file names (and modes) via 'file_paths' and 'modes' key-word
        arguments or a single 'file_path' and 'mode'.

        The access API though is always a list of file paths and modes
        where order is important for associating other features.

        Parameters
        ----------
        file_paths : list of str
            The list of file paths (in order) to use.

        modes : list of str
            The list of mode specs (in order) to use.

        """

        # file paths
        self._file_paths = file_paths

        # modes

        # if modes is None we make modes, from defaults if we have to
        if modes is None:
            # if mode is None set it to the default
            if modes is None:
                mode = self.DEFAULT_MODE

            # if only one mode is given copy it for each file given
            modes = [mode for i in range(len(self._file_paths))]

        for mode in modes:
            if not self._validate_mode(mode):
                raise FileReporterError(f"Invalid file mode: {mode}")

        self._modes = modes

    @property
    def file_paths(self) -> list[Path]:
        """The file paths for this reporter, in order."""
        return self._file_paths

    @property
    def modes(self) -> list[FileMode]:
        """The modes for the files, in order."""
        return self._modes

    def set_path(self, file_idx, path):
        """Set the path for a single indexed file.

        Parameters
        ----------
        file_idx : int
            Index in the listing of files.
        path : str
            The new path to set for this file

        """
        self._paths[file_idx] = path

    # TOREV: shouldn't need this. Can move to using attrs class and
    # evolve if this is an issue elsewhere.

    # @modes.setter
    # def modes(self, modes):
    #     """Setter for the modes.

    #     Parameters
    #     ----------
    #     modes : list of str

    #     """
    #     for i, mode in enumerate(modes):
    #         self.set_mode(i, mode)

    def set_mode(self, file_idx: int, mode: FileMode) -> None:
        """Set the mode for a single indexed file.

        Parameters
        ----------
        file_idx : int
            Index in the listing of files.
        mode : str
            The new mode spec.

        """

        if self._validate_mode(mode):
            self._modes[file_idx] = mode
        else:
            raise FileReporterError(f"Incorrect mode {mode}")

    # def reparametrize(self, file_paths, modes):
    #     """Set the file paths and modes for all files in the reporter.

    #     Parameters
    #     ----------
    #     file_paths : list of str
    #         New file paths for each file, in order.
    #     modes : list of str
    #         New modes for each file, in order.

    #     """

    #     self.file_paths = file_paths
    #     self.modes = modes


class ProgressiveFileReporterABC(FileReporterABC, ABC):
    """Super class for a reporter that will successively overwrite the
    same file over and over again. The base FileReporter really only
    supports creation of file one time.

    """

    def init(self, **kwargs: SimComponentArgs) -> None:

        # because we want to overwrite the file at every cycle we
        # need to change the modes to write with truncate. This allows
        # the file to first be opened in 'x' or 'w-' and check whether
        # the file already exists (say from another run), and warn the
        # user. However, once the file has been created for this run
        # we need to overwrite it many times forcefully.
        logger.info("Initializing ProgressiveFileReporter")

        # go thourgh each file managed by this reporter
        for file_idx, mode in enumerate(self.modes):
            # if the mode is 'x' or 'w-' we check to make sure the file
            # doesn't exist
            if mode in ["x", "w-"]:
                file_path = self.file_paths[file_idx]
                if file_path.exists():
                    raise FileExistsError(f"File exists: {file_path}")

            # now that we have checked if the file exists we set it into
            # overwrite mode
            self.set_mode(file_idx, "w")

    def cleanup(self, **kwargs: SimComponentArgs) -> None:
        logger.info("Nothing to do for ProgressiveFileReporterABC.cleanup.")
