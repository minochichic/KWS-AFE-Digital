"""Regression tests for rawfile parsing, result CSV options, and diagnostics."""

from __future__ import annotations

import inspect
import math
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

from ngspice_channel_sweeper import (
    DEFAULT_TABLE,
    DetectorSettings,
    SweepSettings,
    make_analysis_netlist,
    make_transient_netlist,
    SweeperGUI,
    TransientResult,
    TransientSettings,
    load_channels,
    make_detector_netlist,
    read_ac_raw,
    read_detector_raw,
    read_op_raw,
    read_transient_raw,
    write_channels_csv,
)
from sweeper.components import parse_detector_opamp
from sweeper.netlists import apply_detector_opamp
from sweeper.constants import (
    DEFAULT_DETECTOR_OPAMP,
    DETECTOR_OPAMP_NAMES,
    SCHEMATIC_COMPONENT_POSITIONS,
    SCHEMATIC_VALUE_FONT_SIZE,
)
from sweeper.gui.common import CommonController
from sweeper.gui.detector_tab import DetectorTabController
from sweeper.gui.dc_tab import DcTabController
from sweeper.results import read_ascii_rawfile
from sweeper.simulation import (
    _MAX_STIMULUS_LABEL_CHARS,
    _WINDOWS_MAX_PATH_CHARS,
    _ngspice_failure_message,
    _reject_paths_over_windows_limit,
    _stimulus_directory_name,
)


def _analog_rawfile(rows: tuple[tuple[float, ...], ...]) -> str:
    """Render a five-vector real transient rawfile in ngspice ASCII layout."""

    lines = [
        "Title: fixture",
        "Plotname: Transient Analysis",
        "Flags: real",
        "No. Variables: 5",
        f"No. Points: {len(rows)}",
        "Variables:",
        "\t0\ttime\ttime",
        "\t1\tv(/vin)\tvoltage",
        "\t2\tv(/v_filt)\tvoltage",
        "\t3\tv(/v_env)\tvoltage",
        "\t4\tv(/v_thr)\tvoltage",
        "Values:",
    ]
    for index, row in enumerate(rows):
        lines.append(f"{index}\t{row[0]:.12g}")
        lines.extend(f"\t{value:.12g}" for value in row[1:])
    return "\n".join(lines) + "\n"


_ANALOG_ROWS = (
    (0.000, 0.900, 0.900, 0.900, 0.930),
    (0.050, 0.900, 0.900, 0.900, 0.930),
    (0.100, 0.905, 0.910, 0.920, 0.930),
    (0.150, 0.895, 0.890, 0.940, 0.930),
    (0.350, 0.900, 0.900, 0.900, 0.930),
)


class RawfileParsingTests(unittest.TestCase):
    """The streaming parser must preserve every documented parsing guarantee."""

    def _write(self, directory: str, text: str, name: str = "fixture.raw") -> Path:
        """Write one rawfile fixture and return its path."""

        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_real_rawfile_avoids_boxing_every_sample_as_complex(self) -> None:
        """Real plots store plain floats, which still expose ``.real``."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _analog_rawfile(_ANALOG_ROWS))
            plot = read_ascii_rawfile(path)
        self.assertEqual(len(plot.points), len(_ANALOG_ROWS))
        for point in plot.points:
            for value in point:
                self.assertIsInstance(value, float)
        self.assertEqual(plot.points[2][3], 0.920)
        self.assertEqual(plot.points[2][3].real, 0.920)

    def test_every_sample_matches_the_written_fixture(self) -> None:
        """Chunked token reading must not drop, reorder, or shift samples."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _analog_rawfile(_ANALOG_ROWS))
            plot = read_ascii_rawfile(path)
        self.assertEqual(
            tuple(tuple(point) for point in plot.points),
            _ANALOG_ROWS,
        )

    def test_complex_rawfile_keeps_both_components(self) -> None:
        """AC plots remain complex so magnitude and phase stay correct."""

        text = (
            "Title: fixture\n"
            "Plotname: AC Analysis\n"
            "Flags: complex\n"
            "No. Variables: 2\n"
            "No. Points: 2\n"
            "Variables:\n"
            "\t0\tfrequency\tfrequency\n"
            "\t1\tv(/v_filt)\tvoltage\n"
            "Values:\n"
            "0\t1.0e+01,0.0e+00\n\t2.0e+00,0.0e+00\n"
            "1\t1.0e+02,0.0e+00\n\t0.0e+00,1.0e+00\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            plot = read_ascii_rawfile(path)
            rows = read_ac_raw(path, "/v_filt")
        self.assertEqual(plot.points[0][1], complex(2.0, 0.0))
        self.assertEqual(plot.points[1][1], complex(0.0, 1.0))
        self.assertAlmostEqual(rows[0][0], 10.0)
        self.assertAlmostEqual(rows[0][1], 20.0 * math.log10(2.0))
        self.assertAlmostEqual(rows[0][2], 0.0)
        self.assertAlmostEqual(rows[1][1], 0.0)
        self.assertAlmostEqual(rows[1][2], 90.0)

    def test_fortran_exponents_fall_back_to_the_tolerant_parser(self) -> None:
        """A ``D`` exponent is rejected by ``float`` but must still parse."""

        text = (
            "Title: fixture\n"
            "Plotname: Operating Point\n"
            "Flags: real\n"
            "No. Variables: 2\n"
            "No. Points: 1\n"
            "Variables:\n"
            "\t0\tv(/v_env)\tvoltage\n"
            "\t1\tv(/v_thr)\tvoltage\n"
            "Values:\n"
            "0\t9.0D-01\n\t9.3D-01\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            voltages = read_op_raw(path)
        self.assertAlmostEqual(voltages["/v_env"], 0.90)
        self.assertAlmostEqual(voltages["/v_thr"], 0.93)

    def test_point_count_mismatch_is_still_rejected(self) -> None:
        """A truncated rawfile must fail loudly instead of returning partial data."""

        text = _analog_rawfile(_ANALOG_ROWS).replace(
            f"No. Points: {len(_ANALOG_ROWS)}",
            f"No. Points: {len(_ANALOG_ROWS) + 3}",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            with self.assertRaises(ValueError):
                read_ascii_rawfile(path)

    def test_missing_values_section_is_rejected(self) -> None:
        """A header without a value section cannot silently produce zero points."""

        text = (
            "Title: fixture\n"
            "Flags: real\n"
            "No. Variables: 1\n"
            "No. Points: 1\n"
            "Variables:\n"
            "\t0\ttime\ttime\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            with self.assertRaises(ValueError):
                read_ascii_rawfile(path)

    def test_unknown_node_names_report_the_saved_variables(self) -> None:
        """A misspelled node must explain which vectors the rawfile holds."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _analog_rawfile(_ANALOG_ROWS))
            with self.assertRaises(ValueError) as caught:
                read_transient_raw(
                    path, TransientSettings(venv_node="/not_a_node")
                )
        self.assertIn("/not_a_node", str(caught.exception))
        self.assertIn("v(/v_env)", str(caught.exception))

    def test_column_reader_matches_the_full_plot(self) -> None:
        """Reading only five columns must equal selecting them from every vector."""

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _analog_rawfile(_ANALOG_ROWS))
            plot = read_ascii_rawfile(path)
            transient_rows = read_transient_raw(path, TransientSettings())
            detector_rows = read_detector_raw(path, DetectorSettings())
        expected = [tuple(value.real for value in point) for point in plot.points]
        self.assertEqual(transient_rows, expected)
        self.assertEqual(detector_rows, expected)

    def test_reversed_time_is_rejected(self) -> None:
        """Out-of-order transient samples must not reach the viewer."""

        rows = (
            (0.000, 0.9, 0.9, 0.90, 0.93),
            (0.100, 0.9, 0.9, 0.92, 0.93),
            (0.050, 0.9, 0.9, 0.94, 0.93),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _analog_rawfile(rows))
            with self.assertRaises(ValueError) as caught:
                read_transient_raw(path, TransientSettings())
        self.assertIn("역순", str(caught.exception))


class ResultCsvOptionTests(unittest.TestCase):
    """The duplicated result CSV is optional without changing any other output."""

    def test_transient_settings_expose_a_write_csv_option(self) -> None:
        """Existing library callers keep the CSV unless they opt out."""

        names = {field.name for field in fields(TransientSettings())}
        self.assertIn("write_csv", names)
        self.assertTrue(TransientSettings().write_csv)
        self.assertFalse(TransientSettings(write_csv=False).write_csv)

    def test_detector_settings_expose_a_write_csv_option(self) -> None:
        """Detector shares the transient CSV policy and default."""

        names = {field.name for field in fields(DetectorSettings())}
        self.assertIn("write_csv", names)
        self.assertTrue(DetectorSettings().write_csv)
        self.assertFalse(DetectorSettings(write_csv=False).write_csv)

    def test_disabling_the_csv_does_not_change_validation(self) -> None:
        """The option must not alter any simulation-affecting behaviour."""

        DetectorSettings(write_csv=False).validate()
        TransientSettings(write_csv=False).validate()
        self.assertEqual(
            DetectorSettings(write_csv=False).gate_off_s,
            DetectorSettings().gate_off_s,
        )


class WindowsPathLimitTests(unittest.TestCase):
    """Windows ngspice cannot open paths Python happily creates."""

    def test_stimulus_directory_name_is_short_and_bounded(self) -> None:
        """A long PWL stem must not appear in the generated directory name."""

        stimulus = Path("dataset/yes/004ae714_nohash_0.pwl")
        name = _stimulus_directory_name(1, stimulus)
        self.assertEqual(name, "stim_0001_yes")
        self.assertNotIn("004ae714", name)
        self.assertNotIn("nohash", name)

    def test_stimulus_directory_name_truncates_long_words(self) -> None:
        """Every dataset word yields the same bounded directory length."""

        long_word = Path("dataset/averyverylongword/sample.pwl")
        name = _stimulus_directory_name(42, long_word)
        self.assertEqual(len(name), len("stim_0042_") + _MAX_STIMULUS_LABEL_CHARS)
        self.assertTrue(name.startswith("stim_0042_"))

    def test_stimulus_directory_name_survives_an_unusable_word(self) -> None:
        """A word made only of separators still produces a valid directory."""

        name = _stimulus_directory_name(7, Path("dataset/.../sample.pwl"))
        self.assertEqual(name, "stim_0007")

    def test_paths_within_the_limit_are_accepted(self) -> None:
        """Ordinary result paths must never trip the guard."""

        _reject_paths_over_windows_limit(
            (Path("C:/results/run/ch00/channel_00_params.inc"),)
        )

    def test_over_limit_paths_are_rejected_with_an_actionable_message(
        self,
    ) -> None:
        """The message must name the path, the overage, and the remedy."""

        if sys.platform != "win32":
            self.skipTest("MAX_PATH only constrains Windows")
        offender = Path("C:/") / ("d" * (_WINDOWS_MAX_PATH_CHARS + 20))
        with self.assertRaises(RuntimeError) as caught:
            _reject_paths_over_windows_limit((offender,))
        message = str(caught.exception)
        self.assertIn(str(_WINDOWS_MAX_PATH_CHARS), message)
        self.assertIn("결과 폴더", message)
        self.assertIn("ngspice는 열지 못합니다", message)


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


class DummyButton:
    """Record only the widget state the transient result row toggles."""

    def __init__(self) -> None:
        """Start disabled like the real result radio buttons."""

        self.state = "disabled"

    def configure(self, **options: object) -> None:
        """Apply a state change requested by the controller."""

        if "state" in options:
            self.state = str(options["state"])


class DummyCombo:
    """Capture the dropdown values assigned by the transient controller."""

    def __init__(self) -> None:
        """Start with no stimulus labels."""

        self.values: tuple[str, ...] = ()

    def configure(self, **options: object) -> None:
        """Store newly published stimulus labels."""

        if "values" in options:
            self.values = tuple(options["values"])  # type: ignore[arg-type]


def _transient_result(channel_number: int, stimulus: Path) -> TransientResult:
    """Build one minimal transient result for controller-level tests."""

    channel = load_channels(DEFAULT_TABLE)[channel_number]
    return TransientResult(
        channel=channel,
        stimulus_path=stimulus,
        run_dir=Path("run") / f"ch{channel_number:02d}",
        rows=[(0.0, 0.9, 0.9, 0.9, 0.91), (0.001, 0.9, 0.9, 0.9, 0.91)],
        log_text="",
    )


class LiveTransientResultTests(unittest.TestCase):
    """Finished channels must be viewable while the remaining jobs still run."""

    def _gui(self) -> SweeperGUI:
        """Return a display-free GUI stub wired for transient result install."""

        gui = SweeperGUI.__new__(SweeperGUI)
        gui.transient_result_lookup = {}
        gui.transient_result_stimulus_paths = {}
        gui.transient_result_var = DummyVar("")
        gui.transient_result_channel_var = DummyVar(-1)
        gui.transient_result_combo = DummyCombo()
        gui.transient_result_channel_buttons = {
            number: DummyButton() for number in range(16)
        }
        gui.transient_folder_var = DummyVar(str(Path("dataset")))
        gui.transient_displayed_result = None
        gui.transient_live_rendered = False
        gui.rendered: list[TransientResult] = []
        gui._render_transient_result = gui.rendered.append
        gui._refresh_transient_component_state = lambda: None
        return gui

    def _enabled(self, gui: SweeperGUI) -> list[int]:
        """Return the channel numbers currently clickable in the result row."""

        return [
            number
            for number, button in sorted(
                gui.transient_result_channel_buttons.items()
            )
            if button.state == "normal"
        ]

    def test_first_finished_channel_is_shown_immediately(self) -> None:
        """The user must not wait for the whole job set to see a graph."""

        gui = self._gui()
        stimulus = Path("dataset/yes/sample.pwl")
        gui._install_transient_result(_transient_result(0, stimulus))
        self.assertEqual(self._enabled(gui), [0])
        self.assertEqual(len(gui.rendered), 1)
        self.assertEqual(gui.rendered[0].channel.ch, 0)
        self.assertEqual(int(gui.transient_result_channel_var.get()), 0)

    def test_later_channels_only_become_selectable(self) -> None:
        """A completion must never replace the graph being inspected."""

        gui = self._gui()
        stimulus = Path("dataset/yes/sample.pwl")
        gui._install_transient_result(_transient_result(0, stimulus))
        gui.transient_displayed_result = gui.rendered[0]
        gui._install_transient_result(_transient_result(1, stimulus))
        gui._install_transient_result(_transient_result(2, stimulus))
        self.assertEqual(self._enabled(gui), [0, 1, 2])
        self.assertEqual(len(gui.rendered), 1)
        self.assertEqual(int(gui.transient_result_channel_var.get()), 0)

    def test_every_finished_job_is_retrievable(self) -> None:
        """Each completed channel must be reachable from the result lookup."""

        gui = self._gui()
        stimulus = Path("dataset/yes/sample.pwl")
        for channel_number in (0, 1, 2):
            gui._install_transient_result(
                _transient_result(channel_number, stimulus)
            )
        label = gui.transient_result_var.get()
        for channel_number in (0, 1, 2):
            self.assertIn((label, channel_number), gui.transient_result_lookup)

    def test_distinct_stimuli_get_distinct_labels(self) -> None:
        """Two PWL files sharing a name must not overwrite each other."""

        gui = self._gui()
        first = Path("dataset/yes/sample.pwl")
        second = Path("dataset/no/sample.pwl")
        label_first = gui._transient_result_label(first)
        label_second = gui._transient_result_label(second)
        self.assertNotEqual(label_first, label_second)
        self.assertEqual(gui._transient_result_label(first), label_first)
        self.assertEqual(len(gui.transient_result_stimulus_paths), 2)

    def test_redisplaying_the_same_result_does_not_redraw(self) -> None:
        """Rebuilding the plot would drop the cursors a user already placed."""

        gui = self._gui()
        stimulus = Path("dataset/yes/sample.pwl")
        result = _transient_result(0, stimulus)
        gui._install_transient_result(result)
        gui.transient_displayed_result = gui.rendered[0]
        gui._on_transient_result_channel_changed()
        self.assertEqual(len(gui.rendered), 1)

    def test_selecting_another_finished_channel_redraws(self) -> None:
        """Switching channels must still replace the displayed result."""

        gui = self._gui()
        stimulus = Path("dataset/yes/sample.pwl")
        gui._install_transient_result(_transient_result(0, stimulus))
        gui.transient_displayed_result = gui.rendered[0]
        gui._install_transient_result(_transient_result(1, stimulus))
        gui.transient_result_channel_var.set(1)
        gui._on_transient_result_channel_changed()
        self.assertEqual(len(gui.rendered), 2)
        self.assertEqual(gui.rendered[1].channel.ch, 1)


class DetectorOpampSelectionTests(unittest.TestCase):
    """Only the XU3 model name may change when a different op-amp is chosen."""

    TEMPLATE = (DEFAULT_TABLE.parent / "netlist_template.cir").read_text(
        encoding="utf-8-sig"
    )
    ORIGINAL_NODES = [
        "Net-_U3-+_", "Net-_D1-A_", "1.8V", "GND", "Net-_D1-K_"
    ]

    def _detector_netlist(self, model: str) -> str:
        """Build one detector netlist for the requested op-amp model."""

        return make_detector_netlist(
            self.TEMPLATE,
            Path("params.inc"),
            Path("stimulus.inc"),
            DetectorSettings(),
            model,
        )

    def _xu3(self, netlist: str) -> list[str]:
        """Return the tokens of the detector op-amp's subcircuit call."""

        for line in netlist.splitlines():
            if line.lower().startswith("xu3"):
                return line.split()
        raise AssertionError("XU3 문장이 없습니다.")

    def test_every_model_keeps_the_verified_terminal_order(self) -> None:
        """All three libraries declare IN+, IN-, V+, V-, OUT in that order."""

        for model in DETECTOR_OPAMP_NAMES:
            with self.subTest(model=model):
                parts = self._xu3(self._detector_netlist(model))
                self.assertEqual(parts[1:6], self.ORIGINAL_NODES)
                self.assertEqual(parts[-1], model)

    def test_every_analysis_uses_the_selected_model(self) -> None:
        """AC, OP, and Transient share the circuit, so all must swap XU3 too.

        Only the detector netlist honoured the selection at first, which made a
        model change look like it had no effect on the other tabs.
        """

        for model in DETECTOR_OPAMP_NAMES:
            built = {
                "ac": make_analysis_netlist(
                    self.TEMPLATE, Path("p.inc"), SweepSettings(), "ac", model
                ),
                "op": make_analysis_netlist(
                    self.TEMPLATE, Path("p.inc"), SweepSettings(), "op", model
                ),
                "transient": make_transient_netlist(
                    self.TEMPLATE,
                    Path("p.inc"),
                    Path("s.inc"),
                    TransientSettings(),
                    0.1,
                    model,
                ),
                "detector": self._detector_netlist(model),
            }
            for analysis, netlist in built.items():
                with self.subTest(model=model, analysis=analysis):
                    parts = self._xu3(netlist)
                    self.assertEqual(parts[-1], model)
                    self.assertEqual(parts[1:6], self.ORIGINAL_NODES)

    def test_reduced_template_without_a_detector_stage_still_builds(
        self,
    ) -> None:
        """A template with no XU3 needs no substitution for the default model."""

        minimal = [".title t", "Vsin /vin bias DC 0", "Rload /vin 0 1k"]
        lines, includes = apply_detector_opamp(minimal, DEFAULT_DETECTOR_OPAMP)
        self.assertEqual(lines, minimal)
        self.assertEqual(includes, [])

    def test_reduced_template_rejects_a_non_default_model(self) -> None:
        """Asking for a model the circuit cannot express must fail loudly."""

        minimal = [".title t", "Vsin /vin bias DC 0"]
        with self.assertRaises(ValueError) as caught:
            apply_detector_opamp(minimal, "TLV9042")
        self.assertIn("XU3", str(caught.exception))

    def test_default_model_adds_no_extra_library(self) -> None:
        """OPA379 is already in the template, so its netlist stays unchanged."""

        netlist = self._detector_netlist(DEFAULT_DETECTOR_OPAMP)
        includes = [
            line for line in netlist.splitlines()
            if line.lower().startswith(".include")
        ]
        self.assertEqual(
            sum(1 for line in includes if "OPA379" in line.upper()), 1
        )

    def test_alternate_model_includes_its_own_library(self) -> None:
        """A non-default model needs its library added to the netlist."""

        netlist = self._detector_netlist("TLV9042")
        self.assertIn("TLV9042.lib", netlist)
        self.assertIn("XU3", netlist)

    def test_filter_opamps_are_never_swapped(self) -> None:
        """Only the detector stage is selectable; the filter stays on OPA379."""

        netlist = self._detector_netlist("TLV9041D")
        for reference in ("XU1", "XU2"):
            line = next(
                item for item in netlist.splitlines()
                if item.lower().startswith(reference.lower())
            )
            self.assertEqual(line.split()[-1], "OPA379")

    def test_unknown_model_is_rejected(self) -> None:
        """A typo must fail loudly instead of producing an unusable netlist."""

        with self.assertRaises(ValueError) as caught:
            self._detector_netlist("NOT_A_PART")
        self.assertIn("NOT_A_PART", str(caught.exception))

    def test_factory_csv_loads_with_the_default_model(self) -> None:
        """The read-only CSV predates the column, so it must still load."""

        for channel in load_channels(DEFAULT_TABLE):
            self.assertEqual(channel.detector_opamp, DEFAULT_DETECTOR_OPAMP)

    def test_saved_version_round_trips_the_selected_model(self) -> None:
        """A saved component version must remember each channel's op-amp."""

        channels = load_channels(DEFAULT_TABLE)[:2]
        edited = [
            replace(channels[0], detector_opamp="TLV9042"),
            channels[1],
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "version.csv"
            write_channels_csv(target, edited)
            reloaded = load_channels(target)
        self.assertEqual(reloaded[0].detector_opamp, "TLV9042")
        self.assertEqual(
            reloaded[1].detector_opamp, DEFAULT_DETECTOR_OPAMP
        )

    def test_unknown_model_in_a_version_csv_is_rejected(self) -> None:
        """A hand-edited CSV must not smuggle in an unsupported model."""

        with self.assertRaises(ValueError):
            parse_detector_opamp({"detector_opamp": "LM358"})


class SchematicOverlayTests(unittest.TestCase):
    """Applied values must land on their own placeholder in the schematic."""

    # Measured from circuit_template.png: the "{NAME}" label boxes.
    PLACEHOLDERS = {
        (227, 117): (216, 111, 238, 123),
        (318, 101): (307, 95, 329, 107),
        (310, 187): (299, 181, 321, 193),
        (343, 218): (332, 212, 355, 224),
        (454, 275): (443, 269, 465, 281),
        (613, 303): (602, 297, 625, 309),
        (614, 404): (603, 399, 625, 410),
        (646, 277): (635, 271, 658, 283),
        (787, 101): (776, 95, 798, 107),
        (714, 342): (703, 336, 726, 348),
        (956, 334): (945, 328, 967, 340),
        (1033, 202): (1022, 196, 1045, 208),
        (1033, 346): (1022, 340, 1045, 352),
    }

    def test_every_parameter_has_one_placeholder(self) -> None:
        """The overlay table must cover each editable value exactly once."""

        self.assertEqual(
            len(SCHEMATIC_COMPONENT_POSITIONS), len(self.PLACEHOLDERS)
        )
        for _name, x_pos, y_pos in SCHEMATIC_COMPONENT_POSITIONS:
            self.assertIn((x_pos, y_pos), self.PLACEHOLDERS)

    def test_positions_sit_inside_their_placeholder(self) -> None:
        """A drifted coordinate would draw the value over unrelated symbols."""

        for _name, x_pos, y_pos in SCHEMATIC_COMPONENT_POSITIONS:
            left, top, right, bottom = self.PLACEHOLDERS[(x_pos, y_pos)]
            with self.subTest(position=(x_pos, y_pos)):
                self.assertTrue(left <= x_pos <= right)
                self.assertTrue(top <= y_pos <= bottom)

    def test_overlay_font_stays_small(self) -> None:
        """The placeholders are ~22 px wide, so a large font would overflow."""

        self.assertLessEqual(SCHEMATIC_VALUE_FONT_SIZE, 9)

    def test_schematic_value_keeps_the_compact_suffix(self) -> None:
        """The schematic form must stay short; the table keeps the long units."""

        controller = DcTabController.__new__(DcTabController)
        self.assertEqual(
            DcTabController._schematic_component_value("10.61k"), "10.61k"
        )
        self.assertEqual(
            DcTabController._component_value("10.61k"), "10.61 kΩ"
        )
        del controller


class DummyLabelFrame:
    """Record the ``text`` option a shared component panel is titled with."""

    def __init__(self, text: str = "") -> None:
        """Start with the builder's base title."""

        self.text = text

    def configure(self, **options: object) -> None:
        """Apply a title change requested by the controller."""

        if "text" in options:
            self.text = str(options["text"])

    def cget(self, option: str) -> str:
        """Return the current title for assertions."""

        return self.text if option == "text" else ""


class SharedComponentPanelTests(unittest.TestCase):
    """Each panel edits one channel, so it must say which channel that is."""

    def _gui(self, prefix: str = "detector") -> SweeperGUI:
        """Return a display-free GUI stub wired for one shared panel."""

        gui = SweeperGUI.__new__(SweeperGUI)
        gui.channels = load_channels(DEFAULT_TABLE)
        setattr(gui, f"{prefix}_component_frame", DummyLabelFrame("공유 소자값"))
        setattr(gui, f"{prefix}_component_title", "공유 소자값")
        setattr(gui, f"{prefix}_panel_state_var", DummyVar())
        setattr(gui, f"{prefix}_component_sync_suspended", False)
        setattr(
            gui,
            f"{prefix}_component_vars",
            {
                "R4": DummyVar(),
                "R5": DummyVar(),
                "R6": DummyVar(),
                "C3": DummyVar(),
                "Vthr": DummyVar(),
                "opamp": DummyVar(),
            },
        )
        return gui

    def test_title_names_the_channel_being_edited(self) -> None:
        """Without this the panels silently disagree about their channel."""

        gui = self._gui()
        gui._load_shared_component_fields("detector", 5)
        self.assertEqual(
            gui.detector_component_frame.cget("text"), "공유 소자값 · ch05"
        )
        gui._load_shared_component_fields("detector", 0)
        self.assertEqual(
            gui.detector_component_frame.cget("text"), "공유 소자값 · ch00"
        )

    def test_matching_channel_refreshes_the_fields(self) -> None:
        """An edit to the displayed channel must reach the panel."""

        gui = self._gui()
        gui._load_shared_component_fields("detector", 0)
        gui.channels = [
            replace(channel, r4_kohm=33.0) if channel.ch == 0 else channel
            for channel in gui.channels
        ]
        gui._sync_shared_component_editor("detector", 0, external=True)
        self.assertEqual(gui.detector_component_vars["R4"].get(), "33.00")
        self.assertEqual(gui.detector_panel_state_var.get(), "다른 탭에서 변경됨")

    def test_other_channel_is_reported_but_never_overwrites(self) -> None:
        """Loading the edited channel would switch what this tab edits."""

        gui = self._gui()
        gui._load_shared_component_fields("detector", 0)
        before = gui.detector_component_vars["R4"].get()
        gui._sync_shared_component_editor("detector", 5, external=True)
        self.assertEqual(gui.detector_component_vars["R4"].get(), before)
        self.assertEqual(gui.detector_editing_channel, 0)
        self.assertEqual(
            gui.detector_component_frame.cget("text"), "공유 소자값 · ch00"
        )
        state = gui.detector_panel_state_var.get()
        self.assertIn("ch05", state)
        self.assertIn("ch00", state)

    def test_panel_state_is_separate_from_the_tab_status_line(self) -> None:
        """Transient already owns transient_component_state_var for its tab."""

        source = inspect.getsource(
            CommonController._build_shared_component_editor
        )
        self.assertIn("_panel_state_var", source)
        self.assertNotIn('f"{prefix}_component_state_var"', source)


class DummyListbox:
    """Stand in for a channel selector during headless commit tests."""

    def __init__(self, indices: tuple[int, ...] = ()) -> None:
        """Remember which rows are ticked."""

        self.indices = indices

    def curselection(self) -> tuple[int, ...]:
        """Return the ticked row indices."""

        return self.indices


class BulkComponentApplyTests(unittest.TestCase):
    """The shared panel can write one edit to every selected channel."""

    def _gui(self, ticked: tuple[int, ...], bulk: bool) -> SweeperGUI:
        """Return a display-free GUI stub with a loaded shared panel."""

        gui = SweeperGUI.__new__(SweeperGUI)
        gui.channels = load_channels(DEFAULT_TABLE)
        gui.component_revision = 0
        gui.component_dirty = False
        gui.component_source_path = DEFAULT_TABLE.resolve()
        gui.component_source_label_var = DummyVar()
        gui.component_editor_channel = None
        gui.component_edit_vars = {}
        gui.detector_channel_list = DummyListbox(ticked)
        gui.detector_component_frame = DummyLabelFrame("공유 소자값")
        gui.detector_component_title = "공유 소자값"
        gui.detector_panel_state_var = DummyVar()
        gui.detector_component_sync_suspended = False
        gui.detector_bulk_var = DummyVar(bulk)
        gui.detector_component_vars = {
            "R4": DummyVar(),
            "R5": DummyVar(),
            "R6": DummyVar(),
            "C3": DummyVar(),
            "Vthr": DummyVar(),
            "opamp": DummyVar(),
        }
        gui._load_shared_component_fields("detector", 0)
        gui._record_component_change = lambda *args, **kwargs: None
        return gui

    def test_bulk_off_touches_only_the_panel_channel(self) -> None:
        """The default stays a single-channel edit."""

        gui = self._gui(ticked=(0, 1, 2), bulk=False)
        gui.detector_component_vars["R4"].set("55")
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertEqual(gui._channel_by_number(0).r4_kohm, 55.0)
        self.assertEqual(gui._channel_by_number(1).r4_kohm, 10.0)
        self.assertEqual(gui._channel_by_number(2).r4_kohm, 10.0)

    def test_bulk_on_writes_every_selected_channel(self) -> None:
        """R4/R5/R6/C3 and the op-amp are normally identical across channels."""

        gui = self._gui(ticked=(0, 1, 2), bulk=True)
        gui.detector_component_vars["R4"].set("55")
        gui.detector_component_vars["opamp"].set("TLV9042")
        self.assertTrue(gui._commit_shared_components("detector"))
        for number in (0, 1, 2):
            channel = gui._channel_by_number(number)
            self.assertEqual(channel.r4_kohm, 55.0)
            self.assertEqual(channel.detector_opamp, "TLV9042")
        untouched = gui._channel_by_number(3)
        self.assertEqual(untouched.r4_kohm, 10.0)
        self.assertEqual(untouched.detector_opamp, DEFAULT_DETECTOR_OPAMP)

    def test_bulk_never_overwrites_another_channel_divider(self) -> None:
        """Vthr is per-channel tuning, so only the panel channel may move."""

        gui = self._gui(ticked=(0, 1, 2), bulk=True)
        before = {
            number: gui._channel_by_number(number).r7_kohm
            for number in (1, 2)
        }
        gui.detector_component_vars["R4"].set("55")
        gui.detector_component_vars["Vthr"].set("930.00")
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertAlmostEqual(gui._channel_by_number(0).vthr_v, 0.930, places=3)
        for number, r7_kohm in before.items():
            self.assertEqual(gui._channel_by_number(number).r7_kohm, r7_kohm)

    def test_bulk_counts_as_one_revision(self) -> None:
        """One user edit must invalidate cached results only once."""

        gui = self._gui(ticked=(0, 1, 2, 3), bulk=True)
        before = gui.component_revision
        gui.detector_component_vars["R4"].set("55")
        self.assertTrue(gui._commit_shared_components("detector"))
        self.assertEqual(gui.component_revision, before + 1)

    def test_panel_channel_is_included_even_when_not_ticked(self) -> None:
        """The edited value must never be lost from the panel's own channel."""

        gui = self._gui(ticked=(1, 2), bulk=True)
        gui.detector_component_vars["R4"].set("55")
        self.assertTrue(gui._commit_shared_components("detector"))
        for number in (0, 1, 2):
            self.assertEqual(gui._channel_by_number(number).r4_kohm, 55.0)


class ChannelSelectionFollowTests(unittest.TestCase):
    """Clicking a channel must move the per-channel views of that tab."""

    def test_selector_forwards_selection_changes(self) -> None:
        """The shared builder must offer a hook, not only the run button.

        Without it the Detector list could not tell its component panel which
        channel was clicked, so the panel stayed on the first channel.
        """

        source = inspect.getsource(CommonController._build_channel_selector)
        self.assertIn("on_select", source)
        self.assertIn("_update_run_button", source)

    def test_detector_list_is_wired_to_its_channel_handler(self) -> None:
        """Detector uses the shared selector, so it must pass the hook."""

        source = inspect.getsource(
            DetectorTabController._build_detector_tab
        )
        self.assertIn("on_select=self._on_detector_channel_changed", source)

    def test_dc_list_is_wired_to_its_channel_handler(self) -> None:
        """The DC editor follows the operating-point result channel."""

        source = inspect.getsource(DcTabController._build_dc_tab)
        self.assertIn("on_select=self._on_dc_channel_selection_changed", source)

    def test_dc_selection_ignores_channels_without_a_result(self) -> None:
        """There is no schematic or editor to switch to before a run."""

        gui = SweeperGUI.__new__(SweeperGUI)
        gui.channels = load_channels(DEFAULT_TABLE)
        gui.channel_list = DummyListbox((5,))
        gui.component_editor_channel = 0
        gui.last_op_results = []
        gui.selected_op_channel_var = DummyVar("ch00 (166 Hz)")
        gui._on_dc_channel_selection_changed()
        self.assertEqual(gui.selected_op_channel_var.get(), "ch00 (166 Hz)")


class FailureDiagnosisTests(unittest.TestCase):
    """One failure report format serves AC/OP, Transient, and Detector."""

    def _launch(self, returncode: int = 1) -> SimpleNamespace:
        """Return a launcher result stand-in with both diagnostic log paths."""

        run_dir = Path("run") / "ch00"
        return SimpleNamespace(
            returncode=returncode,
            cwd=run_dir,
            ngspice_log=run_dir / "detector_ngspice.log",
            launcher_log=run_dir / "detector_launcher.log",
        )

    def _long_log(self, *leading: str) -> str:
        """Put the interesting lines first, then plenty of benign output."""

        lines = list(leading)
        lines.extend(f"benign trailing line {index}" for index in range(200))
        return "\n".join(lines)

    def test_error_lines_at_the_start_of_a_long_log_are_reported(self) -> None:
        """A trailing excerpt alone would hide ngspice's real parse errors."""

        ngspice_output = self._long_log(
            "Note: gnd in a subcircuit is not set to 0 automatically",
            "Could not find include file C:/missing/OPA379.LIB",
            "Error on line 26 : xu1 net-_u1-+_ opa379",
        )
        message = _ngspice_failure_message(
            "ch00 Active Detector",
            self._launch(),
            Path("run/ch00/detector.raw"),
            ngspice_output,
            "launcher banner",
        )
        self.assertIn("Could not find include file", message)
        self.assertIn("Error on line 26", message)
        self.assertIn("2행", message)
        self.assertNotIn("benign trailing line 0", message.split("마지막 부분")[0])

    def test_startup_failures_in_the_launcher_log_are_reported(self) -> None:
        """When ngspice cannot open its own -o log the error is on stdout only."""

        launcher_output = self._long_log(
            "[child stdout/stderr follows]",
            "** ngspice-46 : Circuit level simulation program",
            "Batch mode",
            r"C:\deep\path\ch00\tran_ngspice.log: No such file or directory",
        )
        message = _ngspice_failure_message(
            "sample.pwl ch00",
            self._launch(),
            Path("run/ch00/tran.raw"),
            "",
            launcher_output,
        )
        self.assertIn("launcher 로그의 오류 행", message)
        self.assertIn("No such file or directory", message)
        self.assertNotIn("명시적인 오류 행이 없습니다", message)

    def test_ngspice_log_errors_outrank_launcher_log_errors(self) -> None:
        """The simulator's own log is the more specific source when both exist."""

        message = _ngspice_failure_message(
            "sample.pwl ch00",
            self._launch(),
            Path("run/ch00/tran.raw"),
            "Error on line 26 : bad device card",
            "some other failed line",
        )
        self.assertIn("ngspice 로그의 오류 행", message)
        self.assertIn("bad device card", message)

    def test_warning_lines_are_used_when_no_error_line_exists(self) -> None:
        """A warning-only log is still more useful than a bare exit code."""

        message = _ngspice_failure_message(
            "ch00 AC",
            self._launch(),
            Path("run/ch00/ac.raw"),
            self._long_log("Warning: model temperature differs from TNOM"),
            "",
        )
        self.assertIn("경고 행", message)
        self.assertIn("model temperature differs", message)

    def test_severe_symptoms_outrank_the_warning_prefix(self) -> None:
        """A singular matrix is an error-grade symptom even when labelled warning."""

        message = _ngspice_failure_message(
            "ch00 AC",
            self._launch(),
            Path("run/ch00/ac.raw"),
            self._long_log("Warning: singular matrix at node /v_env"),
            "",
        )
        self.assertIn("오류 행", message)
        self.assertIn("singular matrix", message)

    def test_a_log_without_findings_says_so_explicitly(self) -> None:
        """Silence must be reported as silence, not as a missing diagnosis."""

        message = _ngspice_failure_message(
            "ch00 AC",
            self._launch(),
            Path("run/ch00/ac.raw"),
            "Circuit: KiCad schematic\nDoing analysis at TEMP = 27\n",
            "",
        )
        self.assertIn("명시적인 오류 행이 없습니다", message)

    def test_both_log_paths_are_always_included(self) -> None:
        """Detector previously omitted the paths users need to investigate."""

        message = _ngspice_failure_message(
            "ch00 Active Detector",
            self._launch(),
            Path("run/ch00/detector.raw"),
            "Circuit: KiCad schematic\n",
            "",
        )
        self.assertIn("detector_ngspice.log", message)
        self.assertIn("detector_launcher.log", message)

    def test_a_zero_exit_code_without_a_rawfile_is_explained(self) -> None:
        """ngspice can report success yet leave no rawfile behind."""

        message = _ngspice_failure_message(
            "ch00 DC 동작점",
            self._launch(returncode=0),
            Path("run/ch00/op.raw"),
            "Circuit: KiCad schematic\n",
            "",
        )
        self.assertIn("op.raw", message)
        self.assertIn("종료 코드는 0이지만", message)

    def test_pspice_compatibility_hint_reaches_every_runner(self) -> None:
        """The hint used to exist only on the AC/OP path."""

        message = _ngspice_failure_message(
            "sample.pwl ch00",
            self._launch(),
            Path("run/ch00/tran.raw"),
            "Error: no such function 'if'\n",
            "",
        )
        self.assertIn("ngbehavior=pski", message)
        self.assertIn(".spiceinit", message)

    def test_model_include_hint_is_not_raised_by_ordinary_logs(self) -> None:
        """A log merely containing the word include must not trigger the hint."""

        message = _ngspice_failure_message(
            "ch00 AC",
            self._launch(),
            Path("run/ch00/ac.raw"),
            "Note: Compatibility modes selected: ps ki\n"
            "reading include file for the circuit\n",
            "",
        )
        self.assertNotIn("모델 include 경로를 확인하세요", message)


if __name__ == "__main__":
    unittest.main()
