import logging
from wepy.walker import Walker
from wepy.runners.mock import MockRunner, MockState
from wepy.resampling.resamplers.noresampler import NoResampler
from wepy.work_mapper.serial import SerialMapper

from wepy.reporter.dashboard import (
    WalkersSummaryReport,
    WorkerRecord,
    GenSimSectionReport,
    PerformanceSectionReport,
    ResamplerFieldReport,
    ResamplerDashboardSection,
    RunnerDashboardSection,
    BCDashboardSection,
    DashboardReporter,
)

class Test_ResamplerDashboardReporter:
    pass

_LOGGER = logging.getLogger("tests")

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

CYCLE_REPORT_DICT = {
    "cycle_idx" : 1,
    "new_walkers" : [
        Walker(
            MockState(1),
            weight=0.2,
        ),
        Walker(
            MockState(1),
            weight=0.1,
        ),
    ],
    "warp_data" : [],
    "bc_data" : [],
    "progress_data" : {},
    "resampling_data" : [
        [
            {
                "decision_id" : 1,
                "target_idxs" : (0,)
            },
            {
                "decision_id" : 2,
                "target_idxs" : (1, 2)
            },
        ]
    ],
    "resampler_data" : [{}],
    "n_segment_steps": 100,
    "resampled_walkers" : [
        Walker(
            MockState(1),
            weight=0.1,
        ),
        Walker(
            MockState(1),
            weight=0.1,
        ),
        Walker(
            MockState(1),
            weight=0.1,
        ),
    ],
    "runner_precycle_time" : 0.12312,
    "runner_postcycle_time" : 0.234234,
    "sim_manager_segment_overhead_time" : 0.89346,
    "runner_splits_time" : {
        "a" : .234235,
        "b" : .46,
    },
    "worker_segment_times" : {
        0 : [10.213],
        1 : [22.34],
    },
    "cycle_sim_manager_segment_time" : 40.234,
    "cycle_runner_time" : 33.45,
    "cycle_bc_time" : 1.2,
    "cycle_resampling_time" : 6.234,
}


class Test_DashboardReporter:

    def test___init__(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        assert reporter.file_paths == [dash_path]
        assert reporter.modes == ["x"]

    def test_init(self, tmp_path_factory):


        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        assert reporter.file_paths == [dash_path]
        assert reporter.modes == ["x"]
        assert reporter.file_path == dash_path
        assert reporter.mode == "x"

        reporter.init(**SIM_COMPONENTS)
        assert reporter.modes == ["w"]
        assert reporter.mode == "w"

        assert reporter.init_date_time is not None
        assert reporter.total_run_time is not None
        assert reporter.init_sys_time is not None

    def test_calc_walker_summary(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.calc_walker_summary(**CYCLE_REPORT_DICT)

    def test_update_performance_values(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)

        reporter.update_performance_values(**CYCLE_REPORT_DICT)
        
    def test_update_values(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)

        reporter.update_values(**CYCLE_REPORT_DICT)

    def test_write_dashboard(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)

        reporter.write_dashboard("Hello, this is a fake dashboard")
        assert dash_path.exists()
        _LOGGER.info("\n" + dash_path.read_text())

        reporter.write_dashboard("Hello, this is a fake dashboard")
        assert dash_path.exists()
        assert len(dash_path.read_text().strip().split("\n")) == 1
        _LOGGER.info("\n" + dash_path.read_text())

    def test_gen_sim_section(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)

        reporter.update_values(**CYCLE_REPORT_DICT)
        section = reporter.gen_sim_section(**CYCLE_REPORT_DICT)

        _LOGGER.info("\n" + section)

    def test_gen_performance_section(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)

        reporter.update_values(**CYCLE_REPORT_DICT)

        section = reporter.gen_performance_section(**CYCLE_REPORT_DICT)

        _LOGGER.info("\n" + section)
        
    def test_report(self, tmp_path_factory):

        d0 = tmp_path_factory.mktemp("0")
        dash_path = d0 / "main.wepy_dash.org"
        reporter = DashboardReporter(dash_path)

        reporter.init(**SIM_COMPONENTS)
        reporter.report(**CYCLE_REPORT_DICT)

        assert dash_path.exists()

        _LOGGER.info("\n" + dash_path.read_text())
