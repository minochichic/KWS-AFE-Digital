"""Top-level Tk application coordinating independent tab controllers."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from ngspice_runner import resolve_ngspice

from ..constants import (
    DEFAULT_NETLIST,
    DEFAULT_PWL_DATASET,
    DEFAULT_RESULTS,
    DEFAULT_TABLE,
    DEFAULT_WAV_DATASET,
    DIVIDER_NOMINAL_TOTAL_KOHM,
    OP_DEFAULT_SCHEMATIC_FRACTION,
)
from ..models import Channel, ChannelResult, DetectorResult, TransientResult
from ..simulation import DetectorSimulator, Simulator, TransientSimulator
from .ac_tab import AcTabController
from .base import ControllerMethod
from .common import CommonController
from .cursors import InteractivePlotCursor, MetricSelectionOverlay
from .dc_tab import DcTabController
from .detector_tab import DetectorTabController
from .pwl_tab import PwlTabController
from .state import AppState, SharedStateField
from .transient_tab import TransientTabController


class SweeperGUI:
    """Tk shell that owns shared state and coordinates six controllers."""

    default_channels = SharedStateField("default_channels")
    channels = SharedStateField("channels")
    last_results = SharedStateField("last_results")
    last_ac_results = SharedStateField("last_ac_results")
    last_op_results = SharedStateField("last_op_results")
    last_transient_results = SharedStateField("last_transient_results")
    last_detector_result = SharedStateField("last_detector_result")
    detector_result_cache = SharedStateField("detector_result_cache")
    transient_pwl_files = SharedStateField("transient_pwl_files")
    component_source_path = SharedStateField("component_source_path")
    component_dirty = SharedStateField("component_dirty")
    component_editor_channel = SharedStateField("component_editor_channel")
    pending_margin_retune = SharedStateField("pending_margin_retune")
    component_revision = SharedStateField("component_revision")
    component_last_change_source = SharedStateField("component_last_change_source")
    component_last_change_channel = SharedStateField("component_last_change_channel")
    ac_magnitude_unit = SharedStateField("ac_magnitude_unit")
    ac_manual_y_limits = SharedStateField("ac_manual_y_limits")
    transient_result_lookup = SharedStateField("transient_result_lookup")
    transient_result_stimulus_paths = SharedStateField("transient_result_stimulus_paths")
    transient_displayed_result = SharedStateField("transient_displayed_result")
    transient_live_rendered = SharedStateField("transient_live_rendered")
    transient_completed_jobs = SharedStateField("transient_completed_jobs")
    transient_total_jobs = SharedStateField("transient_total_jobs")
    transient_manual_x_limits_ms = SharedStateField("transient_manual_x_limits_ms")
    transient_manual_y_limits_mv = SharedStateField("transient_manual_y_limits_mv")
    transient_y_modes = SharedStateField("transient_y_modes")
    detector_selected_channel = SharedStateField("detector_selected_channel")
    detector_display_mode = SharedStateField("detector_display_mode")
    detector_manual_x_limits_ms = SharedStateField("detector_manual_x_limits_ms")
    detector_manual_y_limits_mv = SharedStateField("detector_manual_y_limits_mv")
    detector_live_rendered = SharedStateField("detector_live_rendered")
    detector_completed_jobs = SharedStateField("detector_completed_jobs")
    detector_total_jobs = SharedStateField("detector_total_jobs")

    def __init__(self, root: object) -> None:
        """Initialize application state, build widgets, and load factory values."""

        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.state = AppState()
        self.common_controller = CommonController(self)
        self.dc_controller = DcTabController(self)
        self.ac_controller = AcTabController(self)
        self.detector_controller = DetectorTabController(self)
        self.transient_controller = TransientTabController(self)
        self.pwl_controller = PwlTabController(self)
        self.root.title("ngspice 16채널 AC / OP / Active Detector / Transient")
        self.root.geometry("1600x1000")
        self.root.minsize(1180, 780)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.simulator: Simulator | None = None
        self.transient_simulator: TransientSimulator | None = None
        self.detector_simulator: DetectorSimulator | None = None
        self.worker: threading.Thread | None = None
        self.transient_worker: threading.Thread | None = None
        self.detector_worker: threading.Thread | None = None
        self.conversion_worker: threading.Thread | None = None
        self.conversion_stop_event = threading.Event()
        self.default_channels: list[Channel] = []
        self.channels: list[Channel] = []
        self.last_results: list[ChannelResult] = []
        self.last_ac_results: list[ChannelResult] = []
        self.last_op_results: list[ChannelResult] = []
        self.last_transient_results: list[TransientResult] = []
        self.last_detector_result: DetectorResult | None = None
        self.detector_result_cache: dict[int, DetectorResult] = {}
        self.transient_pwl_files: list[Path] = []
        self.component_source_path = DEFAULT_TABLE
        self.component_dirty = False
        self.component_editor_channel: int | None = None
        self.component_edit_vars: dict[str, object] = {}
        self.component_auto_update_suspended = False
        self.component_divider_edit_source = "resistors"
        self.component_divider_nominal_total_kohm = (
            DIVIDER_NOMINAL_TOTAL_KOHM
        )
        self.component_vdet_v: float | None = None
        self.dc_component_change_state_var = tk.StringVar(
            value="현재 AppState 소자값"
        )
        self.pending_margin_retune: tuple[int, float] | None = None
        self.metric_selection_overlay: MetricSelectionOverlay | None = None
        self.metric_table = None
        self.selected_metric_channels: set[int] = set()
        self.metric_copy_rows: dict[str, str] = {}
        self.ac_plot_cursors: dict[str, list[InteractivePlotCursor]] = {
            "magnitude": [],
            "phase": [],
        }

        try:
            default_ngspice = resolve_ngspice("")
        except FileNotFoundError:
            default_ngspice = ""
        self.ngspice_var = tk.StringVar(value=default_ngspice)
        self.netlist_var = tk.StringVar(value=str(DEFAULT_NETLIST))
        self.table_var = tk.StringVar(value=str(DEFAULT_TABLE))
        self.component_source_label_var = tk.StringVar(
            value="기본값 (읽기 전용)"
        )
        self.component_vref_var = tk.StringVar(value="")
        self.component_opamp_var = tk.StringVar(value="")
        self.dc_detector_opamp_var = tk.StringVar(value="")
        self.component_margin_var = tk.StringVar(value="")
        self.component_actual_margin_var = tk.StringVar(value="—")
        self.output_var = tk.StringVar(value=str(DEFAULT_RESULTS))
        self.ac_points_var = tk.StringVar(value="100")
        self.ac_start_var = tk.StringVar(value="10")
        self.ac_stop_var = tk.StringVar(value="20000")
        self.output_node_var = tk.StringVar(value="/v_filt")
        self.pspice_compat_var = tk.BooleanVar(value=True)
        self.run_ac_var = tk.BooleanVar(value=True)
        self.run_op_var = tk.BooleanVar(value=True)
        self.mag_x_scale_var = tk.StringVar(value="Decade")
        self.mag_unit_var = tk.StringVar(value="dB")
        # Retain the public v14 widget-variable name as an alias.
        self.mag_y_scale_var = self.mag_unit_var
        self.mag_previous_unit = "dB"
        self.mag_x_min_var = tk.StringVar(value="")
        self.mag_x_max_var = tk.StringVar(value="")
        self.mag_y_min_var = tk.StringVar(value="")
        self.mag_y_max_var = tk.StringVar(value="")
        self.phase_x_scale_var = tk.StringVar(value="Decade")
        self.phase_y_scale_var = tk.StringVar(value="Linear (deg)")
        self.selected_op_channel_var = tk.StringVar(value="")
        self.schematic_zoom_var = tk.DoubleVar(value=1.0)
        self.transient_plot_cursors: list[InteractivePlotCursor] = []
        self.schematic_photo = None
        self.schematic_canvas = None
        self.schematic_result: ChannelResult | None = None
        self.op_paned = None
        self.op_pane_fraction: float | None = OP_DEFAULT_SCHEMATIC_FRACTION
        self.op_canvas_view: tuple[float, float] | None = None
        self.op_detail_notebook = None
        self.op_detail_tabs: dict[str, object] = {}
        self.op_detail_selected_tab = "node"
        self.op_pending_view_state: tuple[
            str,
            float | None,
            tuple[float, float] | None,
        ] | None = None
        self.status_var = tk.StringVar(value="소자 테이블을 불러오는 중...")
        self.pwl_input_var = tk.StringVar(value=str(DEFAULT_WAV_DATASET))
        self.pwl_output_var = tk.StringVar(value=str(DEFAULT_PWL_DATASET))
        self.pwl_overwrite_var = tk.BooleanVar(value=True)
        self.pwl_status_var = tk.StringVar(
            value="WAV 루트와 PWL 출력 폴더를 선택하세요."
        )
        self.transient_folder_var = tk.StringVar(
            value=str(DEFAULT_PWL_DATASET)
        )
        self.transient_output_step_var = tk.StringVar(value="10u")
        self.transient_stop_time_var = tk.StringVar(value="")
        self.transient_max_step_var = tk.StringVar(value="5u")
        self.transient_input_vpp_var = tk.StringVar(value="10")
        # transient.csv duplicates tran.raw, so it stays off unless requested.
        self.transient_write_csv_var = tk.BooleanVar(value=False)
        self.transient_result_var = tk.StringVar(value="")
        self.transient_result_channel_var = tk.IntVar(value=-1)
        self.transient_x_min_var = tk.StringVar(value="")
        self.transient_x_max_var = tk.StringVar(value="")
        self.transient_x_mode_var = tk.StringVar(value="자동")
        self.transient_y_target_var = tk.StringVar(value="v_in")
        self.transient_y_min_var = tk.StringVar(value="")
        self.transient_y_max_var = tk.StringVar(value="")
        self.transient_y_mode_var = tk.StringVar(value="자동")
        self.transient_y_modes = {
            "v_in": "자동",
            "v_filt": "자동",
            "v_env + v_thr": "자동",
        }
        self.transient_range_status_var = tk.StringVar(
            value="X는 세 그래프 공통 · Y는 선택 그래프별로 독립 유지"
        )
        self.transient_status_var = tk.StringVar(
            value="PWL 폴더를 읽고 파일과 채널을 선택하세요."
        )
        self.transient_component_state_var = tk.StringVar(
            value="현재 소자값 상태를 확인하는 중..."
        )
        self.transient_result_lookup: dict[
            tuple[str, int], TransientResult
        ] = {}
        self.transient_result_stimulus_paths: dict[str, Path] = {}
        self.transient_result_channel_buttons: dict[int, object] = {}
        self.transient_axes: dict[str, object] = {}
        self.transient_default_x_limits_ms: tuple[float, float] | None = None
        self.transient_default_y_limits_mv: dict[
            str, tuple[float, float]
        ] = {}
        self.transient_manual_x_limits_ms: tuple[float, float] | None = None
        self.transient_manual_y_limits_mv: dict[
            str, tuple[float, float]
        ] = {}
        self.transient_x_min_entry = None
        self.transient_x_max_entry = None
        self.transient_y_min_entry = None
        self.transient_y_max_entry = None
        self.transient_canvas = None
        self.transient_range_sync_pending = False
        # Edge markers shown when v_thr falls outside the v_env-based range.
        self.transient_vthr_markers: list[object] = []
        self.transient_vthr_series_mv: list[float] = []
        # Live-result progress for the currently running transient job set.
        self.transient_displayed_result: TransientResult | None = None
        self.transient_live_rendered = False
        self.transient_completed_jobs = 0
        self.transient_total_jobs = 0

        self.detector_frequency_var = tk.StringVar(value="1000")
        self.detector_amplitude_var = tk.StringVar(value="10")
        self.detector_amplitude_unit_var = tk.StringVar(value="mVpp")
        self.detector_gate_on_var = tk.StringVar(value="50")
        self.detector_gate_duration_var = tk.StringVar(value="200")
        self.detector_total_time_var = tk.StringVar(value="350")
        self.detector_max_step_var = tk.StringVar(value="5")
        # detector.csv duplicates detector.raw, so it stays off unless requested.
        self.detector_write_csv_var = tk.BooleanVar(value=False)
        self.detector_mode_var = tk.StringVar(value="절대전압")
        self.detector_previous_mode = "절대전압"
        self.detector_x_min_var = tk.StringVar(value="")
        self.detector_x_max_var = tk.StringVar(value="")
        self.detector_y_min_var = tk.StringVar(value="")
        self.detector_y_max_var = tk.StringVar(value="")
        self.detector_status_var = tk.StringVar(
            value="채널과 gated sine 설정을 확인하세요."
        )
        self.detector_result_state_var = tk.StringVar(value="결과 없음")
        self.detector_plot_cursors: list[InteractivePlotCursor] = []
        self.detector_canvas = None
        # Live-result progress for the currently running detector job set.
        self.detector_live_rendered = False
        self.detector_completed_jobs = 0
        self.detector_total_jobs = 0

        self._build()
        self._reload_channels(DEFAULT_TABLE)
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        """Construct the five independent simulation workflow tabs."""

        from tkinter import ttk

        self.workflow_notebook = ttk.Notebook(self.root)
        self.workflow_notebook.pack(fill="both", expand=True)
        self.dc_page = ttk.Frame(self.workflow_notebook)
        self.ac_page = ttk.Frame(self.workflow_notebook)
        self.detector_page = ttk.Frame(self.workflow_notebook)
        self.transient_page = ttk.Frame(self.workflow_notebook)
        self.converter_page = ttk.Frame(self.workflow_notebook)
        self.workflow_notebook.add(self.dc_page, text="DC 동작점")
        self.workflow_notebook.add(self.ac_page, text="AC 시뮬")
        self.workflow_notebook.add(self.detector_page, text="Active Detector")
        self.workflow_notebook.add(self.transient_page, text="Transient 시뮬")
        self.workflow_notebook.add(self.converter_page, text="WAV to PWL 시뮬")
        self.workflow_notebook.bind(
            "<<NotebookTabChanged>>",
            self._on_workflow_tab_changed,
        )

        self._build_dc_tab()
        self._build_ac_tab()
        self._build_detector_tab()
        self._build_transient_tab()
        self._build_converter_tab()
        self._show_empty_results()
        self.workflow_notebook.select(self.dc_page)
        self._update_run_button()

    def _on_workflow_tab_changed(self, _event: object = None) -> None:
        """Redraw retained plot pixels without changing data, ranges, or selection."""

        try:
            selected = self.workflow_notebook.select()
        except Exception:
            return
        if selected == str(self.transient_page):
            self._refresh_transient_component_state()
            canvas = self.transient_canvas
        elif selected == str(self.detector_page):
            self._refresh_detector_component_state()
            canvas = self.detector_canvas
        else:
            return
        if canvas is None:
            return

        def redraw() -> None:
            """Refresh pixels only; preserve result selection and manual limits."""

            try:
                canvas.get_tk_widget().update_idletasks()
                canvas.draw_idle()
            except Exception:
                pass

        self.root.after_idle(redraw)


_CONTROLLER_METHODS = {
    "_arm_plot_cursors": ("common_controller", CommonController),
    "_browse_netlist": ("common_controller", CommonController),
    "_browse_ngspice": ("common_controller", CommonController),
    "_browse_output": ("common_controller", CommonController),
    "_browse_table": ("common_controller", CommonController),
    "_build_channel_selector": ("common_controller", CommonController),
    "_build_shared_component_editor": ("common_controller", CommonController),
    "_shared_component_channel": ("common_controller", CommonController),
    "_shared_commit_targets": ("common_controller", CommonController),
    "_set_shared_component_title": ("common_controller", CommonController),
    "_load_shared_component_fields": ("common_controller", CommonController),
    "_preview_shared_r6": ("common_controller", CommonController),
    "_commit_shared_components": ("common_controller", CommonController),
    "_sync_shared_component_editor": ("common_controller", CommonController),
    "_channels_from_listbox": ("common_controller", CommonController),
    "_show_channel_components": ("common_controller", CommonController),
    "_build_file_settings": ("common_controller", CommonController),
    "_clear_frame": ("common_controller", CommonController),
    "_clear_plot_cursors": ("common_controller", CommonController),
    "_clear_selection": ("common_controller", CommonController),
    "_dispose_cursor_group": ("common_controller", CommonController),
    "_finish_controls": ("common_controller", CommonController),
    "_open_results": ("common_controller", CommonController),
    "_path_row": ("common_controller", CommonController),
    "_poll_events": ("common_controller", CommonController),
    "_reload_channels": ("common_controller", CommonController),
    "_record_component_change": ("common_controller", CommonController),
    "_render_results": ("common_controller", CommonController),
    "_reset_component_defaults": ("common_controller", CommonController),
    "_save_component_version_from_gui": ("common_controller", CommonController),
    "_select_all": ("common_controller", CommonController),
    "_selected_analyses": ("common_controller", CommonController),
    "_settings": ("common_controller", CommonController),
    "_show_empty_results": ("common_controller", CommonController),
    "_start_run": ("common_controller", CommonController),
    "_stop_run": ("common_controller", CommonController),
    "_update_run_button": ("common_controller", CommonController),
    "_apply_component_editor_from_gui": ("dc_controller", DcTabController),
    "_build_dc_tab": ("dc_controller", DcTabController),
    "_build_op_detail_tables": ("dc_controller", DcTabController),
    "_canvas_boxed_text": ("dc_controller", DcTabController),
    "_capture_operating_point_view_state": ("dc_controller", DcTabController),
    "_change_schematic_zoom": ("dc_controller", DcTabController),
    "_channel_by_number": ("dc_controller", DcTabController),
    "_commit_component_editor": ("dc_controller", DcTabController),
    "_component_editor_has_changes": ("dc_controller", DcTabController),
    "_component_rows": ("dc_controller", DcTabController),
    "_component_value": ("dc_controller", DcTabController),
    "_schematic_component_value": ("dc_controller", DcTabController),
    "_drag_schematic": ("dc_controller", DcTabController),
    "_draw_operating_point_schematic": ("dc_controller", DcTabController),
    "_format_component_resistance_fields": ("dc_controller", DcTabController),
    "_lookup_voltage": ("dc_controller", DcTabController),
    "_on_dc_channel_selection_changed": ("dc_controller", DcTabController),
    "_on_op_channel_changed": ("dc_controller", DcTabController),
    "_on_op_detail_tab_changed": ("dc_controller", DcTabController),
    "_prepare_margin_retune": ("dc_controller", DcTabController),
    "_preview_r6_from_r4_r5": ("dc_controller", DcTabController),
    "_preview_margin_from_vthr": ("dc_controller", DcTabController),
    "_preview_resistors_from_margin": ("dc_controller", DcTabController),
    "_refresh_dc_model_summary": ("dc_controller", DcTabController),
    "_preview_vref_from_resistors": ("dc_controller", DcTabController),
    "_redraw_current_schematic": ("dc_controller", DcTabController),
    "_refresh_operating_point_tab": ("dc_controller", DcTabController),
    "_render_operating_point": ("dc_controller", DcTabController),
    "_restore_operating_point_view_state": ("dc_controller", DcTabController),
    "_set_schematic_zoom": ("dc_controller", DcTabController),
    "_set_dc_component_change_status": ("dc_controller", DcTabController),
    "_start_schematic_pan": ("dc_controller", DcTabController),
    "_stop_schematic_pan": ("dc_controller", DcTabController),
    "_sync_dc_component_editor": ("dc_controller", DcTabController),
    "_zoom_schematic_with_wheel": ("dc_controller", DcTabController),
    "_build_ac_tab": ("ac_controller", AcTabController),
    "_build_metrics_table": ("ac_controller", AcTabController),
    "_clear_ac_selection": ("ac_controller", AcTabController),
    "_copy_metric_rows": ("ac_controller", AcTabController),
    "_embed_figure": ("ac_controller", AcTabController),
    "_format_metric": ("ac_controller", AcTabController),
    "_magnitude_axis_limits": ("ac_controller", AcTabController),
    "_on_magnitude_unit_changed": ("ac_controller", AcTabController),
    "_parse_axis_limits": ("ac_controller", AcTabController),
    "_plot_controls": ("ac_controller", AcTabController),
    "_refresh_magnitude_tab": ("ac_controller", AcTabController),
    "_refresh_phase_tab": ("ac_controller", AcTabController),
    "_render_magnitude": ("ac_controller", AcTabController),
    "_render_phase": ("ac_controller", AcTabController),
    "_reset_magnitude_axis_limits": ("ac_controller", AcTabController),
    "_restore_magnitude_y_limits": ("ac_controller", AcTabController),
    "_select_all_ac": ("ac_controller", AcTabController),
    "_sync_metric_selection": ("ac_controller", AcTabController),
    "_store_magnitude_y_limits": ("ac_controller", AcTabController),
    "_toggle_metric_table_row": ("ac_controller", AcTabController),
    "_apply_detector_axis_ranges": ("detector_controller", DetectorTabController),
    "_build_detector_tab": ("detector_controller", DetectorTabController),
    "_detector_axis_limits": ("detector_controller", DetectorTabController),
    "_detector_channel_number": ("detector_controller", DetectorTabController),
    "_selected_detector_channels": ("detector_controller", DetectorTabController),
    "_select_all_detector_channels": ("detector_controller", DetectorTabController),
    "_clear_detector_channels": ("detector_controller", DetectorTabController),
    "_detector_settings": ("detector_controller", DetectorTabController),
    "_embed_detector_figure": ("detector_controller", DetectorTabController),
    "_finish_detector_controls": ("detector_controller", DetectorTabController),
    "_install_detector_result": ("detector_controller", DetectorTabController),
    "_on_detector_channel_changed": ("detector_controller", DetectorTabController),
    "_on_detector_mode_changed": ("detector_controller", DetectorTabController),
    "_populate_detector_channels": ("detector_controller", DetectorTabController),
    "_refresh_detector_component_state": ("detector_controller", DetectorTabController),
    "_refresh_detector_plot": ("detector_controller", DetectorTabController),
    "_render_detector_result": ("detector_controller", DetectorTabController),
    "_reset_detector_axis_ranges": ("detector_controller", DetectorTabController),
    "_restore_detector_axis_ranges": ("detector_controller", DetectorTabController),
    "_start_detector_run": ("detector_controller", DetectorTabController),
    "_stop_detector_run": ("detector_controller", DetectorTabController),
    "_store_detector_axis_ranges": ("detector_controller", DetectorTabController),
    "_update_detector_run_button": ("detector_controller", DetectorTabController),
    "_apply_transient_x_range": ("transient_controller", TransientTabController),
    "_annotate_offscreen_vthr": ("transient_controller", TransientTabController),
    "_apply_transient_y_range": ("transient_controller", TransientTabController),
    "_browse_transient_folder": ("transient_controller", TransientTabController),
    "_build_transient_range_controls": ("transient_controller", TransientTabController),
    "_build_transient_tab": ("transient_controller", TransientTabController),
    "_clear_transient_channels": ("transient_controller", TransientTabController),
    "_clear_transient_files": ("transient_controller", TransientTabController),
    "_embed_multi_axes_figure": ("transient_controller", TransientTabController),
    "_finish_transient_controls": ("transient_controller", TransientTabController),
    "_install_transient_result": ("transient_controller", TransientTabController),
    "_install_transient_results": ("transient_controller", TransientTabController),
    "_transient_result_label": ("transient_controller", TransientTabController),
    "_on_transient_channel_selection_changed": ("transient_controller", TransientTabController),
    "_on_transient_result_changed": ("transient_controller", TransientTabController),
    "_on_transient_result_channel_changed": ("transient_controller", TransientTabController),
    "_on_transient_x_mode_changed": ("transient_controller", TransientTabController),
    "_on_transient_y_mode_changed": ("transient_controller", TransientTabController),
    "_on_transient_y_target_changed": ("transient_controller", TransientTabController),
    "_populate_transient_channels": ("transient_controller", TransientTabController),
    "_refresh_transient_component_state": ("transient_controller", TransientTabController),
    "_refresh_transient_pwl_files": ("transient_controller", TransientTabController),
    "_refresh_transient_result_channel_buttons": ("transient_controller", TransientTabController),
    "_render_transient_result": ("transient_controller", TransientTabController),
    "_schedule_transient_range_sync": ("transient_controller", TransientTabController),
    "_select_all_transient_channels": ("transient_controller", TransientTabController),
    "_select_all_transient_files": ("transient_controller", TransientTabController),
    "_show_transient_setting_basis": ("transient_controller", TransientTabController),
    "_start_transient_run": ("transient_controller", TransientTabController),
    "_stop_transient_run": ("transient_controller", TransientTabController),
    "_sync_transient_range_fields": ("transient_controller", TransientTabController),
    "_transient_settings": ("transient_controller", TransientTabController),
    "_transient_stimulus_label": ("transient_controller", TransientTabController),
    "_update_transient_range_entry_states": ("transient_controller", TransientTabController),
    "_update_transient_run_button": ("transient_controller", TransientTabController),
    "_validated_axis_pair": ("transient_controller", TransientTabController),
    "_browse_pwl_input": ("pwl_controller", PwlTabController),
    "_browse_pwl_output": ("pwl_controller", PwlTabController),
    "_build_converter_tab": ("pwl_controller", PwlTabController),
    "_finish_pwl_controls": ("pwl_controller", PwlTabController),
    "_scan_wav_tree": ("pwl_controller", PwlTabController),
    "_start_pwl_conversion": ("pwl_controller", PwlTabController),
    "_stop_pwl_conversion": ("pwl_controller", PwlTabController),
}

def _install_controller_methods() -> None:
    """Expose each composed controller method through the legacy GUI class."""

    for method_name, (attribute, controller_type) in _CONTROLLER_METHODS.items():
        setattr(
            SweeperGUI,
            method_name,
            ControllerMethod(attribute, controller_type, method_name),
        )


_install_controller_methods()
