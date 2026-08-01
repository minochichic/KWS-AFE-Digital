"""Regression tests for AC magnitude units and the Active Detector workflow."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import ngspice_channel_sweeper as compatibility_module
from sweeper.components import load_channels
from sweeper.constants import (
    C3_ALLOWED_NF,
    C3_ALLOWED_NF_LABELS,
    DEFAULT_NETLIST,
    DEFAULT_TABLE,
)
from sweeper.gui.ac_tab import AcTabController
from sweeper.gui.common import CommonController
from sweeper.gui.app import SweeperGUI
from sweeper.gui.detector_tab import DetectorTabController
from sweeper.gui.state import AppState
from sweeper.models import DetectorResult, DetectorSettings
from sweeper.netlists import (
    find_input_voltage_source,
    make_detector_netlist,
    make_gated_sine_pairs,
    make_gated_sine_source_include,
)
from sweeper.results import (
    detector_ac_comparison_rows,
    measure_detector_response,
)
from sweeper.simulation import DetectorSimulator
from sweeper.values import (
    db_to_linear,
    linear_magnitude_auto_limits,
    linear_to_db,
)


class DummyVar:
    """Provide Tk-compatible get/set behavior for display-free controller tests."""

    def __init__(self, value: object = "") -> None:
        """Store the initial value."""

        self.value = value

    def get(self) -> object:
        """Return the stored value."""

        return self.value

    def set(self, value: object) -> None:
        """Replace the stored value."""

        self.value = value


def _response_rows() -> list[tuple[float, float, float, float, float]]:
    """Return a response whose threshold crossings have exact interpolants."""

    return [
        (0.000, 0.90, 0.80, 0.90, 0.95),
        (0.040, 0.90, 0.80, 0.90, 0.95),
        (0.050, 0.90, 0.80, 0.90, 0.95),
        (0.060, 0.91, 0.81, 0.92, 0.95),
        (0.070, 0.89, 0.79, 0.94, 0.95),
        (0.250, 0.90, 0.80, 0.94, 0.95),
        (0.260, 0.90, 0.80, 0.92, 0.95),
        (0.270, 0.90, 0.80, 0.90, 0.95),
        (0.300, 0.90, 0.80, 0.90, 0.95),
    ]


class AcMagnitudeUnitTests(unittest.TestCase):
    """Verify exact dB/Linear conversion and unit-specific range retention."""

    def test_db_linear_round_trip_and_small_values_are_not_clipped(self) -> None:
        """Values below -20 dB must remain distinct exact voltage ratios."""

        self.assertAlmostEqual(db_to_linear(-20.0), 0.1, places=15)
        self.assertAlmostEqual(db_to_linear(-40.0), 0.01, places=15)
        self.assertAlmostEqual(db_to_linear(-80.0), 0.0001, places=15)
        self.assertLess(db_to_linear(-40.0), 0.1)
        for magnitude_db in (-120.0, -40.0, -20.0, 0.0, 17.25):
            self.assertAlmostEqual(
                linear_to_db(db_to_linear(magnitude_db)),
                magnitude_db,
                places=12,
            )

    def test_linear_half_power_and_zero_based_auto_range(self) -> None:
        """The -3.0103 dB level equals peak over sqrt(2) on a linear Y axis."""

        half_power_db = -10.0 * math.log10(2.0)
        self.assertAlmostEqual(
            db_to_linear(half_power_db),
            1.0 / math.sqrt(2.0),
            places=15,
        )
        self.assertEqual(
            linear_magnitude_auto_limits((0.0001, 0.01, 2.0)),
            (0.0, 2.1),
        )

    def test_db_and_linear_manual_ranges_are_saved_independently(self) -> None:
        """Switching units restores each unit's unconverted manual Y text."""

        app = SimpleNamespace(
            ac_manual_y_limits={"dB": ("", ""), "Linear": ("", "")},
            mag_unit_var=DummyVar("dB"),
            mag_y_min_var=DummyVar("-55"),
            mag_y_max_var=DummyVar("12"),
        )
        controller = AcTabController(app)
        controller._store_magnitude_y_limits("dB")
        app.mag_y_min_var.set("0")
        app.mag_y_max_var.set("3.5")
        controller._store_magnitude_y_limits("Linear")
        controller._restore_magnitude_y_limits("dB")
        self.assertEqual(app.mag_y_min_var.get(), "-55")
        self.assertEqual(app.mag_y_max_var.get(), "12")
        controller._restore_magnitude_y_limits("Linear")
        self.assertEqual(app.mag_y_min_var.get(), "0")
        self.assertEqual(app.mag_y_max_var.get(), "3.5")

    def test_magnitude_renderer_never_uses_a_logarithmic_y_axis(self) -> None:
        """Only the retained X Decade control may request logarithmic scaling."""

        source = inspect.getsource(AcTabController._render_magnitude)
        self.assertIn('magnitude_unit == "Linear"', source)
        self.assertIn("linear_magnitude_auto_limits", source)
        self.assertNotIn("set_yscale", source)


class DetectorNetlistTests(unittest.TestCase):
    """Verify the gated source and complete latest full-channel circuit."""

    def test_gated_sine_has_exact_vpp_and_zero_pre_and_post_gate(self) -> None:
        """Sampled source peaks must span Vpp without moving gate timestamps."""

        settings = DetectorSettings()
        pairs = make_gated_sine_pairs(settings)
        active = [
            voltage
            for time_s, voltage in pairs
            if settings.gate_on_s <= time_s <= settings.gate_off_s
        ]
        self.assertAlmostEqual(
            max(active) - min(active),
            settings.input_vpp_v,
            places=14,
        )
        self.assertEqual(pairs[0], (0.0, 0.0))
        self.assertEqual(pairs[-1], (settings.total_time_s, 0.0))
        self.assertIn((settings.gate_off_s, 0.0), pairs)

    def test_detector_replaces_v4_at_vin_and_keeps_full_circuit(self) -> None:
        """The filter, detector, comparator, models, and source terminals remain."""

        template = DEFAULT_NETLIST.read_text(encoding="utf-8")
        settings = DetectorSettings()
        source_name, positive, negative = find_input_voltage_source(template)
        self.assertEqual((source_name, positive, negative), (
            "V4",
            "/vin",
            "Net-_V3-Pad1_",
        ))
        stimulus = make_gated_sine_source_include(
            source_name,
            positive,
            negative,
            settings,
        )
        self.assertIn("V4 /vin Net-_V3-Pad1_ PWL(", stimulus)
        self.assertNotIn("V4 /v_filt", stimulus)
        netlist = make_detector_netlist(
            template,
            Path("/tmp/params.inc"),
            Path("/tmp/detector_stimulus.inc"),
            settings,
        )
        for circuit_fragment in (
            "XU1 Net-_U1-+_ Net-_U1--_ 1.8v GND /v_filt OPA379",
            "XU3 Net-_U3-+_ Net-_D1-A_ 1.8V GND Net-_D1-K_ OPA379",
            "D1 Net-_D1-A_ Net-_D1-K_ Dbat54wt1",
            "D2 Net-_D1-K_ /v_env Dbat54wt1",
            "XU4 /v_env /v_thr 1.8V GND /v_comp LPV7215",
            ".save v(/vin) v(/v_filt) v(/v_env) v(/v_thr)",
        ):
            self.assertIn(circuit_fragment, netlist)
        self.assertNotIn("SIN(", netlist)

    def test_timestep_warning_and_both_editable_presets_are_explicit(self) -> None:
        """The 20-point warning and quick/paper durations remain distinguishable."""

        quick = DetectorSettings.quick_200ms_preset()
        paper = DetectorSettings.paper_2020_preset()
        self.assertAlmostEqual(quick.gate_duration_s, 0.200)
        self.assertAlmostEqual(paper.gate_duration_s, 1.500)
        self.assertIsNone(quick.timestep_warning())
        coarse = DetectorSettings(maximum_step_s=100e-6)
        warning = coarse.timestep_warning()
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertIn("20 points/cycle", warning)
        self.assertIn("50 us", warning)
        with self.assertRaisesRegex(ValueError, "한 주기"):
            DetectorSettings(gate_duration_s=0.0005).validate()

    @unittest.skipIf(os.name == "nt", "POSIX shebang test double")
    def test_fake_ngspice_detector_run_parses_and_writes_latest_nodes(self) -> None:
        """The isolated simulator emits canonical CSV without running device models."""

        original_hash = hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest()
        fake = Path(__file__).with_name("fake_ngspice.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = DetectorSimulator(log=lambda _text: None).run(
                load_channels(DEFAULT_TABLE)[0],
                DEFAULT_NETLIST,
                Path(directory),
                str(fake),
                DetectorSettings(),
                component_revision=9,
            )
            self.assertEqual(result.component_revision, 9)
            self.assertEqual(len(result.rows), 8)
            header = (result.run_dir / "detector.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertEqual(
                header,
                "time_s,vin_v,v_filt_v,v_env_v,v_thr_v",
            )
            source = (result.run_dir / "detector_stimulus.inc").read_text(
                encoding="ascii"
            )
            self.assertIn("V4 /vin Net-_V3-Pad1_ PWL(", source)
        self.assertEqual(
            hashlib.sha256(DEFAULT_TABLE.read_bytes()).hexdigest(),
            original_hash,
        )


class DetectorMeasurementTests(unittest.TestCase):
    """Verify baseline removal and ordered linearly interpolated crossings."""

    def test_ac_comparison_removes_each_required_pre_gate_dc_value(self) -> None:
        """Waveform amplitudes and time remain untouched except specified offsets."""

        rows = [
            (0.00, 0.90, 0.80, 0.70, 0.95),
            (0.04, 0.92, 0.82, 0.72, 0.95),
            (0.05, 0.93, 0.83, 0.74, 0.95),
            (0.06, 0.89, 0.78, 0.76, 0.95),
        ]
        converted, baselines = detector_ac_comparison_rows(rows, 0.05)
        self.assertEqual(baselines, (0.91, 0.81, 0.71, 0.95))
        self.assertEqual([row[0] for row in converted], [row[0] for row in rows])
        self.assertAlmostEqual(converted[0][1], -0.01)
        self.assertAlmostEqual(converted[1][2], 0.01)
        self.assertAlmostEqual(converted[2][3], 0.03)
        self.assertAlmostEqual(converted[0][4], 0.24)

    def test_headroom_rise_and_fall_use_linear_interpolation(self) -> None:
        """First ordered pairs after Gate ON/OFF return exact crossing times."""

        settings = DetectorSettings(
            gate_on_s=0.05,
            gate_duration_s=0.20,
            total_time_s=0.30,
        )
        measured = measure_detector_response(
            _response_rows(),
            settings,
            0.91,
            0.93,
        )
        self.assertTrue(measured.success)
        self.assertAlmostEqual(measured.headroom_v, 0.02)
        self.assertAlmostEqual(measured.rise_low_s or 0.0, 0.055)
        self.assertAlmostEqual(measured.rise_high_s or 0.0, 0.065)
        self.assertAlmostEqual(measured.fall_high_s or 0.0, 0.255)
        self.assertAlmostEqual(measured.fall_low_s or 0.0, 0.265)
        self.assertAlmostEqual(measured.rise_time_s or 0.0, 0.010)
        self.assertAlmostEqual(measured.fall_time_s or 0.0, 0.010)

    def test_invalid_levels_and_missing_crossings_report_failures(self) -> None:
        """Bad level order raises, while absent crossings return no invented time."""

        settings = DetectorSettings(
            gate_on_s=0.05,
            gate_duration_s=0.20,
            total_time_s=0.30,
        )
        with self.assertRaisesRegex(ValueError, "Vhigh"):
            measure_detector_response(_response_rows(), settings, 0.93, 0.91)
        flat = [
            (time_s, 0.9, 0.9, 0.9, 0.95)
            for time_s in (0.0, 0.04, 0.05, 0.10, 0.25, 0.30)
        ]
        measured = measure_detector_response(flat, settings, 0.91, 0.93)
        self.assertFalse(measured.success)
        self.assertIsNone(measured.rise_time_s)
        self.assertIsNone(measured.fall_time_s)
        self.assertIn("Vlow", measured.rise_reason)
        self.assertIn("Vhigh", measured.fall_reason)

    def test_ripple_broken_first_attempt_is_skipped_for_next_valid_pair(self) -> None:
        """A low crossing invalidated by ripple does not pair with a later high."""

        settings = DetectorSettings(
            gate_on_s=0.05,
            gate_duration_s=0.20,
            total_time_s=0.30,
        )
        rows = [
            (0.00, 0.9, 0.9, 0.900, 0.95),
            (0.05, 0.9, 0.9, 0.900, 0.95),
            (0.06, 0.9, 0.9, 0.920, 0.95),
            (0.07, 0.9, 0.9, 0.905, 0.95),
            (0.08, 0.9, 0.9, 0.915, 0.95),
            (0.09, 0.9, 0.9, 0.940, 0.95),
            (0.25, 0.9, 0.9, 0.940, 0.95),
            (0.26, 0.9, 0.9, 0.920, 0.95),
            (0.27, 0.9, 0.9, 0.900, 0.95),
            (0.30, 0.9, 0.9, 0.900, 0.95),
        ]
        measured = measure_detector_response(rows, settings, 0.91, 0.93)
        self.assertAlmostEqual(measured.rise_low_s or 0.0, 0.075)
        self.assertAlmostEqual(measured.rise_high_s or 0.0, 0.086)


class DetectorStateTests(unittest.TestCase):
    """Verify AppState sharing, revision behavior, stale data, and range isolation."""

    def _headless_component_gui(self) -> SweeperGUI:
        """Build enough shared GUI state to exercise a Detector component commit."""

        channel = load_channels(DEFAULT_TABLE)[0]
        gui = SweeperGUI.__new__(SweeperGUI)
        gui.channels = [channel]
        gui.default_channels = [channel]
        gui.component_revision = 4
        gui.component_source_path = DEFAULT_TABLE.resolve()
        gui.component_dirty = False
        gui.component_source_label_var = DummyVar()
        gui.component_editor_channel = channel.ch
        gui.component_auto_update_suspended = False
        gui.component_edit_vars = {
            "R4": DummyVar(f"{channel.r4_kohm:.2f}"),
            "R5": DummyVar(f"{channel.r5_kohm:.2f}"),
            "R6": DummyVar(f"{channel.r6_kohm:.2f}"),
            "C3": DummyVar(f"{channel.c3_nf:g}"),
        }
        gui.component_vref_var = DummyVar()
        gui.component_opamp_var = DummyVar(channel.detector_opamp)
        gui.component_margin_var = DummyVar("")
        gui.dc_detector_opamp_var = DummyVar(channel.detector_opamp)
        gui.component_vdet_v = None
        gui.schematic_canvas = None
        gui.schematic_result = None
        gui.dc_component_change_state_var = DummyVar()
        gui.detector_selected_channel = channel.ch
        gui.detector_opamp_var = DummyVar(channel.detector_opamp)
        gui.detector_component_sync_suspended = False
        gui.detector_component_vars = {
            "R4": DummyVar("20"),
            "R5": DummyVar("60"),
            "R6": DummyVar("8.25"),
            "C3": DummyVar("100"),
        }
        gui.detector_panel_state_var = DummyVar()
        gui.detector_editing_channel = channel.ch
        gui.detector_component_vars["Vthr"] = DummyVar(
            f"{channel.vthr_v * 1000.0:.2f}"
        )
        gui.detector_component_vars["opamp"] = DummyVar(channel.detector_opamp)
        gui.detector_result_state_var = DummyVar()
        gui.detector_result_cache = {}
        return gui

    def test_component_edits_sync_tabs_and_increment_once_each(self) -> None:
        """R4/R5 automation, R6 override, and C3 update share one AppState."""

        gui = self._headless_component_gui()
        gui._preview_shared_r6("detector")
        self.assertEqual(gui.detector_component_vars["R6"].get(), "15.00")
        before = gui.component_revision
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertEqual(gui.component_revision, before + 1)
        self.assertEqual(gui._channel_by_number(0).r6_kohm, 15.0)
        self.assertEqual(gui.detector_panel_state_var.get(), "변경됨")
        self.assertEqual(
            gui.dc_component_change_state_var.get(),
            "다른 탭에서 변경됨",
        )
        self.assertFalse(gui._commit_shared_components("detector"))
        self.assertEqual(gui.component_revision, before + 1)

        gui.detector_component_vars["R6"].set("17.34")
        self.assertTrue(gui._commit_shared_components("detector"))
        direct = gui._channel_by_number(0)
        self.assertEqual((direct.r4_kohm, direct.r5_kohm), (20.0, 60.0))
        self.assertEqual(direct.r6_kohm, 17.34)

        gui.detector_component_vars["R4"].set("30")
        gui._preview_shared_r6("detector")
        self.assertEqual(gui.detector_component_vars["R6"].get(), "20.00")
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertEqual(gui._channel_by_number(0).r6_kohm, 20.0)

        gui.detector_component_vars["C3"].set("22")
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertEqual(gui._channel_by_number(0).c3_nf, 22.0)

    def test_c3_list_is_exact_and_detector_combo_is_read_only(self) -> None:
        """Only the requested 25 E-series nanofarad values are exposed."""

        self.assertEqual(
            C3_ALLOWED_NF,
            (
                1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7,
                5.6, 6.8, 8.2, 10.0, 12.0, 15.0, 18.0, 22.0,
                27.0, 33.0, 39.0, 47.0, 56.0, 68.0, 82.0, 100.0,
            ),
        )
        self.assertEqual(C3_ALLOWED_NF_LABELS[0], "1.0")
        self.assertEqual(C3_ALLOWED_NF_LABELS[12:], (
            "10", "12", "15", "18", "22", "27", "33",
            "39", "47", "56", "68", "82", "100",
        ))
        source = inspect.getsource(
            CommonController._build_shared_component_editor
        )
        self.assertIn('state="readonly"', source)
        self.assertIn("C3_ALLOWED_NF_LABELS", source)

    def test_detector_and_transient_state_are_independent(self) -> None:
        """Detector selection/cache/ranges cannot overwrite PWL transient state."""

        state = AppState()
        transient_path = Path("speech/sample.pwl")
        state.transient_pwl_files = [transient_path]
        state.transient_manual_x_limits_ms = (1.0, 2.0)
        state.transient_manual_y_limits_mv["v_in"] = (-5.0, 5.0)
        channel = load_channels(DEFAULT_TABLE)[0]
        detector = DetectorResult(
            channel=channel,
            settings=DetectorSettings(),
            component_revision=3,
            run_dir=Path("detector"),
            rows=_response_rows(),
            log_text="",
        )
        state.detector_selected_channel = channel.ch
        state.detector_result_cache[channel.ch] = detector
        state.detector_manual_x_limits_ms["절대전압"] = ("0", "300")
        self.assertEqual(state.transient_pwl_files, [transient_path])
        self.assertEqual(state.transient_manual_x_limits_ms, (1.0, 2.0))
        self.assertEqual(
            state.transient_manual_y_limits_mv["v_in"],
            (-5.0, 5.0),
        )
        detector_source = inspect.getsource(DetectorTabController)
        for transient_attribute in (
            "transient_pwl_files",
            "last_transient_results",
            "transient_result_lookup",
            "transient_manual_x_limits_ms",
            "transient_manual_y_limits_mv",
            "transient_plot_cursors",
        ):
            self.assertNotIn(transient_attribute, detector_source)

    def test_detector_result_stale_flag_uses_component_revision(self) -> None:
        """A cached result clearly becomes stale after one component edit."""

        channel = load_channels(DEFAULT_TABLE)[0]
        result = DetectorResult(
            channel=channel,
            settings=DetectorSettings(),
            component_revision=7,
            run_dir=Path("detector"),
            rows=_response_rows(),
            log_text="",
        )
        self.assertFalse(result.is_stale(7))
        self.assertTrue(result.is_stale(8))

    def test_absolute_and_ac_comparison_ranges_are_independent(self) -> None:
        """Each display mode restores its own X and Y range text."""

        app = SimpleNamespace(
            detector_manual_x_limits_ms={
                "절대전압": ("", ""),
                "AC 비교": ("", ""),
            },
            detector_manual_y_limits_mv={
                "절대전압": ("", ""),
                "AC 비교": ("", ""),
            },
            detector_mode_var=DummyVar("절대전압"),
            detector_x_min_var=DummyVar("0"),
            detector_x_max_var=DummyVar("350"),
            detector_y_min_var=DummyVar("850"),
            detector_y_max_var=DummyVar("980"),
        )
        controller = DetectorTabController(app)
        controller._store_detector_axis_ranges("절대전압")
        app.detector_x_min_var.set("40")
        app.detector_x_max_var.set("270")
        app.detector_y_min_var.set("-50")
        app.detector_y_max_var.set("80")
        controller._store_detector_axis_ranges("AC 비교")
        controller._restore_detector_axis_ranges("절대전압")
        self.assertEqual(app.detector_x_min_var.get(), "0")
        self.assertEqual(app.detector_y_min_var.get(), "850")
        controller._restore_detector_axis_ranges("AC 비교")
        self.assertEqual(app.detector_x_min_var.get(), "40")
        self.assertEqual(app.detector_y_min_var.get(), "-50")


class CompatibilityAndDocumentationTests(unittest.TestCase):
    """Keep public imports, tab ordering, schematic assets, and docs aligned."""

    def test_detector_public_symbols_and_controller_method_are_available(self) -> None:
        """The expanded API adds Detector symbols without removing v14 imports."""

        for name in (
            "DetectorSettings",
            "DetectorResult",
            "DetectorSimulator",
            "make_detector_netlist",
            "read_detector_raw",
            "SweeperGUI",
        ):
            self.assertTrue(hasattr(compatibility_module, name), name)
        self.assertEqual(
            Path(inspect.getsourcefile(SweeperGUI._build_detector_tab) or "").name,
            "detector_tab.py",
        )

    def test_top_level_tabs_have_the_requested_five_item_order(self) -> None:
        """Active Detector remains between AC and Transient with DC selected."""

        source = inspect.getsource(SweeperGUI._build)
        labels = (
            'text="DC 동작점"',
            'text="AC 시뮬"',
            'text="Active Detector"',
            'text="Transient 시뮬"',
            'text="WAV to PWL 시뮬"',
        )
        positions = [source.index(label) for label in labels]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("workflow_notebook.select(self.dc_page)", source)

    def test_latest_schematic_asset_and_overlay_nodes_are_present(self) -> None:
        """The PDF-derived image and all five canonical node coordinates exist."""

        from PIL import Image

        image_path = DEFAULT_NETLIST.with_name("circuit_template.png")
        with Image.open(image_path) as image:
            self.assertEqual(image.size, (1280, 590))
        source = inspect.getsource(
            SweeperGUI._draw_operating_point_schematic
        )
        for node in ("/vin", "/v_filt", "/v_env", "/v_thr", "/v_comp"):
            self.assertIn(f'("{node}"', source)

    def test_production_and_docs_contain_no_legacy_voltage_net_names(self) -> None:
        """All runtime, template, README, and architecture net names are current."""

        root = DEFAULT_NETLIST.parent
        paths = [
            DEFAULT_NETLIST,
            root / "README_ko.md",
            root / "ARCHITECTURE_ko.md",
            *sorted((root / "sweeper").rglob("*.py")),
        ]
        legacy = (
            "/v_" + "filt_out",
            "/v_" + "detect_out",
            "/v_" + "comp_out",
            "/v_" + "ref",
        )
        failures: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for name in legacy:
                if name in text:
                    failures.append(f"{path.name}: {name}")
        self.assertEqual(failures, [])

    def test_architecture_documents_detector_data_flow(self) -> None:
        """The Korean architecture file explains the independent Detector path."""

        architecture = DEFAULT_NETLIST.with_name("ARCHITECTURE_ko.md").read_text(
            encoding="utf-8"
        )
        for phrase in (
            "DetectorSettings",
            "DetectorSimulator",
            "DetectorResult",
            "component_revision",
            "detector_tab.py",
            "Vheadroom = Vhigh - Vlow",
        ):
            self.assertIn(phrase, architecture)


if __name__ == "__main__":
    unittest.main()
