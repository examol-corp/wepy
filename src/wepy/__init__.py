"""Top-level package."""

# Local Modules
from .__about__ import __version__ as __version__
from .analysis.contig_tree import (
    BaseContigTree,
    Contig,
    ContigTree,
)
from .analysis.network import (
    BaseMacroStateNetwork,
    MacroStateNetwork,
)
from .analysis.network_layouts.layout_graph import LayoutGraph
from .analysis.parents import (
    ParentForest,
    ancestors,
    net_parent_table,
    parent_cycle_discontinuities,
    parent_panel,
    parent_table_discontinuities,
    resampling_panel,
    sliding_window,
)
from .analysis.profiles import (
    ContigTreeProfiler,
    contigtrees_bin_edges,
    cumulative_partitions,
    free_energy_profile,
)
from .analysis.rates import (
    calc_warp_rate,
    contig_warp_rates,
)
from .boundary_conditions.boundary import BoundaryConditions
from .hdf5 import WepyHDF5
from .monitor import Monitor
from .reporter.base import Reporter
from .reporter.dashboard import (
    BCDashboardSection,
    DashboardReporter,
    ResamplerDashboardSection,
    RunnerDashboardSection,
)
from .reporter.hdf5 import WepyHDF5Reporter
from .reporter.openmm import OpenMMRunnerDashboardSection
from .reporter.restree import ResTreeReporter
from .reporter.revo.dashboard import REVODashboardSection
from .resampling.decisions.clone_merge import MultiCloneMergeDecision
from .resampling.decisions.decision import BaseDecisionABC, BaseDecisionRecord
from .resampling.decisions.no_decision import NoDecision
from .resampling.distances.base import Distance
from .resampling.distances.mock import MockDistance
from .resampling.distances.simple import XYDistanceState, XYEuclideanDistance
from .resampling.resamplers.noresampler import NoResampler, NoResamplerFactory
from .resampling.resamplers.resampler import Resampler
from .resampling.resamplers.revo import REVOResampler, REVOResamplerFactory
from .resampling.resamplers.wexplore import WExploreResampler, WExploreResamplerFactory
from .runners.mock import (
    MockRunner,
    MockRunnerFactory,
    MockState,
)
from .runners.openmm.logger import (
    EnergyLoggingReporter,
    EnergyLoggingReporterFactory,
    HeartBeatLoggingReporter,
    HeartBeatLoggingReporterFactory,
    SamplingTimeIntervalLoggingReporter,
    StepIntervalLoggingReporter,
    UnitCellLoggingReporter,
    UnitCellLoggingReporterFactory,
)
from .runners.openmm.reporter import OpenMMReporter, OpenMMReporterNextReport
from .runners.openmm.runner import (
    OpenMMRunner,
    OpenMMRunnerFactory,
    DEFAULT_OPENMM_REPORTER_FACTORIES,
)
from .runners.openmm.state import (
    OPENMM_DEFAULT_UNITS,
    OpenMMState,
    OpenMMStateWrapper,
)
from .runners.runner import NoRunner, NoRunnerFactory, Runner
from .sim_manager import (
    Manager,
    ResamplerFactory,
    RunnerFactory,
    WorkMapperFactory,
)
from .util.json_top import (
    json_top_atom_count,
    json_top_atom_df,
    json_top_chain_df,
    json_top_residue_df,
    json_top_subset,
)
from .util.mdtraj import (
    json_to_mdtraj_topology,
    mdtraj_to_json_topology,
    traj_fields_to_mdtraj,
)
from .walker import (
    Walker,
    WalkerState,
    WalkerStateBox,
)
from .work_mapper.base import WorkMapper
from .work_mapper.openmm.proc_pool import (
    OpenMMProcPoolWorkMapper,
    OpenMMProcPoolWorkMapperFactory,
)
from .work_mapper.openmm.serial import (
    OpenMMSerialWorkMapper,
    OpenMMSerialWorkMapperFactory,
)
from .work_mapper.serial import SerialMapper, SerialMapperFactory

__author__ = "Samuel D. Lotz"
__email__ = "samuel.lotz@salotz.info"

__all__ = [
    "LayoutGraph",
    "__version__",
    "WepyHDF5",
    "Monitor",
    "Manager",
    "Reporter",
    "ResamplerDashboardSection",
    "RunnerDashboardSection",
    "BCDashboardSection",
    "OpenMMRunnerDashboardSection",
    "DashboardReporter",
    "WepyHDF5Reporter",
    "OpenMMReporterDashboardSection",
    "ResTreeReporter",
    "REVODashboardSection",
    "BaseDecisionRecord",
    "BaseDecisionABC",
    "Walker",
    "WalkerState",
    "WalkerStateBox",
    "NoDecision",
    "MultiCloneMergeDecision",
    "NoResampler",
    "NoResamplerFactory",
    "Resampler",
    "REVOResampler",
    "WExploreResampler",
    "Distance",
    "MockDistance",
    "XYDistanceState",
    "XYEuclideanDistance",
    "Runner",
    "NoRunner",
    "MockState",
    "MockRunner",
    "MockRunnerFactory",
    "OpenMMReporter",
    "OpenMMReporterNextReport",
    "StepIntervalLoggingReporter",
    "SamplingTimeIntervalLoggingReporter",
    "HeartBeatLoggingReporter",
    "HeartBeatLoggingReporterFactory",
    "EnergyLoggingReporter",
    "EnergyLoggingReporterFactory",
    "UnitCellLoggingReporter",
    "UnitCellLoggingReporterFactory",
    "OpenMMStateWrapper",
    "OpenMMState",
    "OpenMMRunner",
    "OpenMMRunnerFactory",
    "json_top_atom_df",
    "json_top_residue_df",
    "json_top_chain_df",
    "json_top_atom_count",
    "json_top_subset",
    "mdtraj_to_json_topology",
    "json_to_mdtraj_topology",
    "traj_fields_to_mdtraj",
    "BaseContigTree",
    "ContigTree",
    "Contig",
    "BaseMacroStateNetwork",
    "MacroStateNetwork",
    "resampling_panel",
    "parent_panel",
    "net_parent_table",
    "parent_table_discontinuities",
    "sliding_window",
    "ParentForest",
    "ancestors",
    "parent_cycle_discontinuities",
    "cumulative_partitions",
    "free_energy_profile",
    "contigtrees_bin_edges",
    "ContigTreeProfiler",
    "calc_warp_rate",
    "contig_warp_rates",
    "BoundaryConditions",
    "WorkMapper",
    "SerialMapper",
    "OpenMMSerialWorkMapper",
    "OpenMMSerialWorkMapperFactory",
    "OpenMMProcPoolWorkMapper",
    "OpenMMProcPoolWorkMapperFactory",
    "ResamplerFactory",
    "WorkMapperFactory",
    "RunnerFactory",
    "REVOResamplerFactory",
    "NoRunnerFactory",
    "SerialMapperFactory",
    "OPENMM_DEFAULT_UNITS",
    "WExploreResamplerFactory",
]
