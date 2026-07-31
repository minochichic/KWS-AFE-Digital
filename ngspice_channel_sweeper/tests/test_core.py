"""Regression tests for component versioning, metrics, and analysis plumbing."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import os
import tempfile
import unittest
import wave
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from ngspice_channel_sweeper import (
    DEFAULT_TABLE,
    AcMetrics,
    ChannelResult,
    InteractivePlotCursor,
    MetricSelectionOverlay,
    SweepSettings,
    analysis_directives,
    apply_component_updates,
    build_analysis_jobs,
    calculate_ac_metrics,
    data_occupancy_limits,
    divider_resistors_for_vref,
    divider_vref_v,
    find_input_voltage_source,
    format_margin_mv,
    format_metric_copy_line,
    format_resistance_kohm,
    format_voltage_mv,
    load_channels,
    make_pwl_source_include,
    make_transient_netlist,
    parse_time_seconds,
    parallel_resistance_kohm,
    read_transient_raw,
    recommended_maximum_step_s,
    round_resistance_kohm,
    save_component_version,
    scale_pwl_pairs_to_vpp,
    transient_rows_to_ms_mv,
    TransientResult,
    TransientSettings,
    SweeperGUI,
    TransientSimulator,
    vref_from_margin_v,
    voltage_margin_v,
    write_channels_csv,
)
from wav_pwl import (
    convert_wav_tree,
    normalize_zero_dc_vpp,
    read_pwl_data,
)


def file_sha256(path: Path) -> str:
    """Return a stable digest used to prove the factory CSV is untouched."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class DummyVar:
    """Small get/set stand-in for Tk variables in headless GUI-state tests."""

    def __init__(self, value: object = "") -> None:
        """Store one mutable value for headless state tests."""

        self.value = value

    def get(self) -> object:
        """Return the current value."""

        return self.value

    def set(self, value: object) -> None:
        """Replace the current value."""

        self.value = value


class DummyWidget:
    """Capture ttk ``configure`` options without requiring a display server."""

    def __init__(self) -> None:
        """Initialize an empty option dictionary."""

        self.options: dict[str, object] = {}

    def configure(self, **options: object) -> None:
        """Record the most recently configured values."""

        self.options.update(options)


class ComponentVersionTests(unittest.TestCase):
    """Protect the factory table and verify lossless 16-channel versions."""

    def test_version_save_never_changes_default(self) -> None:
        """Editing and saving must create a new CSV and preserve the default hash."""

        before = file_sha256(DEFAULT_TABLE)
        channels = load_channels(DEFAULT_TABLE)
        changed = apply_component_updates(
            channels[0],
            {
                "RA": "13.5",
                "R1": str(channels[0].r1_kohm),
                "R2": str(channels[0].r2_kohm),
                "R4": str(channels[0].r4_kohm),
                "R5": str(channels[0].r5_kohm),
                "R6": str(channels[0].r6_kohm),
                "R7": "480",
                "R8": "520",
                "C1": str(channels[0].c_nf),
                "C3": str(channels[0].c3_nf),
            },
        )
        channels[0] = changed
        with tempfile.TemporaryDirectory() as directory:
            saved = save_component_version(channels, Path(directory))
            reloaded = load_channels(saved)
            self.assertEqual(len(reloaded), 16)
            self.assertAlmostEqual(reloaded[0].ra_kohm, 13.5)
            self.assertAlmostEqual(reloaded[0].vthr_v, 0.936)
        self.assertEqual(file_sha256(DEFAULT_TABLE), before)

    def test_direct_default_write_is_rejected(self) -> None:
        """The low-level CSV writer must refuse the factory path itself."""

        with self.assertRaises(PermissionError):
            write_channels_csv(DEFAULT_TABLE, load_channels(DEFAULT_TABLE))

    def test_invalid_component_text_is_rejected(self) -> None:
        """Non-positive component values must fail before simulation."""

        channel = load_channels(DEFAULT_TABLE)[0]
        values = {
            "RA": "0",
            "R1": "1",
            "R2": "1",
            "R4": "1",
            "R5": "1",
            "R6": "1",
            "R7": "1",
            "R8": "1",
            "C1": "1",
            "C3": "1",
        }
        with self.assertRaises(ValueError):
            apply_component_updates(channel, values)

    def test_editable_margin_derives_vref_and_hundredth_kohm_resistors(
        self,
    ) -> None:
        """Vmargin controls Vref while auto R7/R8 use two decimal places."""

        channel = load_channels(DEFAULT_TABLE)[0]
        values = {
            "RA": str(channel.ra_kohm),
            "R1": str(channel.r1_kohm),
            "R2": str(channel.r2_kohm),
            "R4": str(channel.r4_kohm),
            "R5": str(channel.r5_kohm),
            "R6": str(channel.r6_kohm),
            "R7": str(channel.r7_kohm),
            "R8": str(channel.r8_kohm),
            "C1": str(channel.c_nf),
            "C3": str(channel.c3_nf),
        }
        changed = apply_component_updates(
            channel,
            values,
            requested_margin_mv="13.4847",
            reference_vdet_v=0.9034847,
        )
        self.assertEqual(changed.r7_kohm, round(changed.r7_kohm, 2))
        self.assertEqual(changed.r8_kohm, round(changed.r8_kohm, 2))
        self.assertAlmostEqual(
            divider_vref_v(changed.r7_kohm, changed.r8_kohm),
            changed.vthr_v,
            places=12,
        )
        self.assertAlmostEqual(changed.vthr_v, 0.9169694, delta=0.0002)

    def test_margin_recalculation_uses_stable_nominal_divider_total(
        self,
    ) -> None:
        """Repeated margin commits must not accumulate resistor-rounding drift."""

        channel = load_channels(DEFAULT_TABLE)[0]
        values = {
            "RA": str(channel.ra_kohm),
            "R1": str(channel.r1_kohm),
            "R2": str(channel.r2_kohm),
            "R4": str(channel.r4_kohm),
            "R5": str(channel.r5_kohm),
            "R6": str(channel.r6_kohm),
            "R7": str(channel.r7_kohm),
            "R8": str(channel.r8_kohm),
            "C1": str(channel.c_nf),
            "C3": str(channel.c3_nf),
        }
        nominal_total = channel.r7_kohm + channel.r8_kohm
        first = apply_component_updates(
            channel,
            values,
            requested_margin_mv="13.4847",
            reference_vdet_v=0.9034847,
            divider_nominal_total_kohm=nominal_total,
        )
        values["R7"] = str(first.r7_kohm)
        values["R8"] = str(first.r8_kohm)
        second = apply_component_updates(
            first,
            values,
            requested_margin_mv="13.4847",
            reference_vdet_v=0.9034847,
            divider_nominal_total_kohm=nominal_total,
        )
        self.assertEqual(
            (first.r7_kohm, first.r8_kohm),
            (second.r7_kohm, second.r8_kohm),
        )

    def test_divider_rounding_does_not_force_exact_nominal_sum(self) -> None:
        """Independent 0.01 kΩ rounding may legitimately move the total."""

        r7, r8 = divider_resistors_for_vref(
            1.8 * 123.455 / 1000.0,
            nominal_total_kohm=1000.0,
        )
        self.assertAlmostEqual(r7, 876.55)
        self.assertAlmostEqual(r8, 123.46)
        self.assertAlmostEqual(r7 + r8, 1000.01)

    def test_every_resistance_is_rounded_and_formatted_to_two_decimals(
        self,
    ) -> None:
        """Editor commits, display text, and SPICE values share 0.01 kΩ precision."""

        channel = load_channels(DEFAULT_TABLE)[0]
        values = {
            "RA": "12.345",
            "R1": "16.555",
            "R2": "100.004",
            "R4": "10.126",
            "R5": "47.124",
            "R6": "8.246",
            "R7": "490.345",
            "R8": "509.655",
            "C1": "76.567",
            "C3": "100.123",
        }
        changed = apply_component_updates(channel, values)
        self.assertEqual(changed.ra_kohm, 12.35)
        self.assertEqual(changed.r1_kohm, 16.56)
        self.assertEqual(changed.r2_kohm, 100.00)
        self.assertEqual(changed.r4_kohm, 10.13)
        self.assertEqual(changed.r5_kohm, 47.12)
        self.assertEqual(changed.r6_kohm, 8.25)
        self.assertEqual(changed.r7_kohm, 490.35)
        self.assertEqual(changed.r8_kohm, 509.66)
        self.assertEqual(format_resistance_kohm(10.0), "10.00")
        self.assertEqual(round_resistance_kohm(1.235), 1.24)
        self.assertEqual(changed.spice_parameters()["R2"], "100.00k")

    def test_r6_parallel_default_is_two_decimal_but_can_be_overridden(
        self,
    ) -> None:
        """R4/R5 calculate the initial R6 field without constraining later R6."""

        self.assertEqual(parallel_resistance_kohm(10.0, 47.0), 8.25)
        preview = inspect.getsource(SweeperGUI._preview_r6_from_r4_r5)
        build = inspect.getsource(SweeperGUI._build_op_detail_tables)
        self.assertIn('component_edit_vars["R4"]', preview)
        self.assertIn('component_edit_vars["R5"]', preview)
        self.assertIn('component_edit_vars["R6"].set', preview)
        self.assertNotIn('component_edit_vars["R6"].get', preview)
        self.assertIn('if key in {"R4", "R5"}', build)
        self.assertIn("trace_add", build)

    def test_divider_orientation_and_vref_range_are_validated(self) -> None:
        """R8 is the lower leg; invalid rail-valued Vref must be rejected."""

        r7, r8 = divider_resistors_for_vref(0.9)
        self.assertAlmostEqual(r7, 500.0)
        self.assertAlmostEqual(r8, 500.0)
        with self.assertRaises(ValueError):
            divider_resistors_for_vref(1.8)

    def test_vref_is_vdet_plus_margin(self) -> None:
        """Positive Vmargin places Vref above the detector DC level."""

        self.assertAlmostEqual(
            vref_from_margin_v(0.9034847, 0.0134847),
            0.9169694,
        )
        with self.assertRaises(ValueError):
            vref_from_margin_v(0.9, 0.0)
        with self.assertRaises(ValueError):
            vref_from_margin_v(0.9, -0.001)
        with self.assertRaises(ValueError):
            vref_from_margin_v(0.9, 1.0)

    def test_first_new_op_vdet_can_refine_margin_once(self) -> None:
        """A detector-changing edit gets one R7/R8 correction from fresh Vdet."""

        channel = load_channels(DEFAULT_TABLE)[0]
        gui = SweeperGUI.__new__(SweeperGUI)
        gui.pending_margin_retune = (channel.ch, 0.020)
        gui.channels = [channel]
        gui.component_editor_channel = None
        gui.component_edit_vars = {}
        gui.component_dirty = False
        result = ChannelResult(
            channel=channel,
            run_dir=Path("."),
            ac_rows=[],
            op_voltages={"/v_detect_out": 0.940},
            log_text="",
        )
        updated = gui._prepare_margin_retune([result])
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.r7_kohm, round(updated.r7_kohm, 2))
        self.assertEqual(updated.r8_kohm, round(updated.r8_kohm, 2))
        self.assertAlmostEqual(updated.vthr_v, 0.960, delta=0.0002)
        self.assertIsNone(gui.pending_margin_retune)


class MetricTests(unittest.TestCase):
    """Verify the measured peak/Q math and copy-friendly formatting."""

    def test_ideal_second_order_bandpass_q(self) -> None:
        """A dense ideal Q=5 response should be recovered near 1 kHz."""

        f0 = 1000.0
        expected_q = 5.0
        rows = []
        for index in range(2001):
            frequency = 100.0 * 10.0 ** (2.0 * index / 2000.0)
            ratio = frequency / f0
            magnitude = (ratio / expected_q) / math.sqrt(
                (1.0 - ratio * ratio) ** 2
                + (ratio / expected_q) ** 2
            )
            rows.append(
                (frequency, 20.0 * math.log10(magnitude), 0.0)
            )
        metric = calculate_ac_metrics(rows)
        self.assertIsNotNone(metric)
        assert metric is not None
        self.assertAlmostEqual(metric.center_hz, f0, delta=0.2)
        self.assertIsNotNone(metric.q)
        assert metric.q is not None
        self.assertAlmostEqual(metric.q, expected_q, delta=0.01)

    def test_metric_copy_line_is_one_tsv_line(self) -> None:
        """The UI clipboard form must remain one tab-separated line per channel."""

        channel = load_channels(DEFAULT_TABLE)[0]
        metric = AcMetrics(166.1, 6.02, 120.0, 220.0, 1.661, "정상")
        line = format_metric_copy_line(channel, metric)
        self.assertNotIn("\n", line)
        self.assertEqual(len(line.split("\t")), 4)
        self.assertIn("f0=166.1 Hz", line)
        self.assertIn("gain=6.02 dB", line)
        self.assertIn("Q=1.661", line)

    def test_voltage_display_uses_millivolts(self) -> None:
        """A 0.9174 V internal value must display as 917.4 mV."""

        self.assertEqual(format_voltage_mv(0.9174), "917.4 mV")

    def test_voltage_margin_is_vref_minus_vdet(self) -> None:
        """The DC readout must preserve the sign of threshold headroom."""

        margin = voltage_margin_v(0.9061619, 0.9127446)
        self.assertAlmostEqual(margin, 0.0065827)
        self.assertEqual(format_margin_mv(margin), "+6.5827 mV")

    def test_multiple_metric_cursors_can_coexist(self) -> None:
        """Two selected rows should create two independent cursor triples."""

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        first = axes.plot([100.0, 200.0], [0.0, 1.0], color="blue")[0]
        second = axes.plot([300.0, 400.0], [1.0, 0.0], color="red")[0]
        overlay = MetricSelectionOverlay(
            canvas,
            axes,
            {
                0: (
                    first,
                    AcMetrics(166.0, 1.0, 120.0, 220.0, 1.66, "정상"),
                    1.0,
                ),
                1: (
                    second,
                    AcMetrics(350.0, 1.0, 300.0, 400.0, 3.5, "정상"),
                    1.0,
                ),
            },
        )
        overlay.show_channels([0, 1])
        self.assertEqual(len(overlay.artists), 6)
        overlay.clear()
        self.assertEqual(overlay.artists, [])

    def test_general_cursor_requires_button_arm_and_keeps_multiple_markers(
        self,
    ) -> None:
        """Only armed motion previews a line; repeated clicks keep markers."""

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        line = axes.plot([1.0, 2.0], [3.0, 4.0], label="trace")[0]
        canvas.draw()
        pixel_x, pixel_y = axes.transData.transform((1.0, 3.0))
        event = SimpleNamespace(
            button=1,
            inaxes=axes,
            xdata=1.0,
            ydata=3.0,
            x=pixel_x,
            y=pixel_y,
        )
        cursor = InteractivePlotCursor(canvas, axes, [line])
        cursor._on_motion(event)
        self.assertFalse(cursor.preview_vertical.get_visible())
        cursor._on_click(event)
        self.assertEqual(cursor.artists, [])
        cursor.arm()
        cursor._on_motion(event)
        self.assertTrue(cursor.preview_vertical.get_visible())
        self.assertEqual(tuple(cursor.preview_vertical.get_xdata()), (1.0, 1.0))
        cursor._on_click(event)
        self.assertEqual(len(cursor.artists), 1)
        self.assertFalse(cursor.preview_vertical.get_visible())
        cursor.arm()
        cursor._on_motion(event)
        cursor._on_click(event)
        self.assertEqual(len(cursor.artists), 2)
        cursor.clear()
        self.assertEqual(cursor.artists, [])
        cursor.dispose()
        self.assertTrue(cursor._disposed)


class AnalysisArchitectureTests(unittest.TestCase):
    """Keep AC/OP jobs isolated while transient uses the same launcher contract."""

    def test_registered_jobs_have_isolated_paths(self) -> None:
        """AC and OP must resolve to separate netlists, rawfiles, and log stems."""

        jobs = build_analysis_jobs(Path("run"), 3, ("ac", "op"))
        self.assertEqual([job.spec.key for job in jobs], ["ac", "op"])
        self.assertEqual(jobs[0].netlist_path.name, "channel_03_ac.cir")
        self.assertEqual(jobs[1].netlist_path.name, "channel_03_op.cir")
        self.assertNotEqual(jobs[0].raw_path, jobs[1].raw_path)

    def test_analysis_directives_are_separated(self) -> None:
        """Each job receives only its own analysis statements."""

        settings = SweepSettings()
        ac_text = "\n".join(analysis_directives(settings, "ac")).lower()
        op_text = "\n".join(analysis_directives(settings, "op")).lower()
        self.assertIn(".ac dec", ac_text)
        self.assertNotIn(".op", ac_text)
        self.assertIn(".op", op_text)
        self.assertNotIn(".ac ", op_text)


class WavPwlTests(unittest.TestCase):
    """Verify speech normalization, folder mirroring, and zero-voltage tail."""

    @staticmethod
    def _write_test_wav(path: Path, samples: list[int], rate: int = 8_000) -> None:
        """Write a tiny mono 16-bit PCM fixture using only the standard library."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(
            int(sample).to_bytes(2, byteorder="little", signed=True)
            for sample in samples
        )
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(payload)

    def test_zero_dc_and_exact_ten_millivolt_pp(self) -> None:
        """Mean removal and range scaling must satisfy both requested targets."""

        values, stats = normalize_zero_dc_vpp([-4.0, -1.0, 3.0, 8.0])
        self.assertAlmostEqual(math.fsum(values) / len(values), 0.0, places=15)
        self.assertAlmostEqual(max(values) - min(values), 0.010, places=15)
        self.assertAlmostEqual(float(stats["output_vpp_v"]), 0.010, places=15)

    def test_tree_conversion_mirrors_word_and_ends_at_zero(self) -> None:
        """A word/WAV input must become word/PWL with one explicit 0 V tail."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "speech_commands_for_sim"
            output_root = source_root / "pwl"
            wav_path = source_root / "go" / "sample.wav"
            self._write_test_wav(wav_path, [-1000, -250, 500, 1000])
            converted = convert_wav_tree(source_root, output_root)
            self.assertEqual(len(converted), 1)
            pwl_path = output_root / "go" / "sample.pwl"
            self.assertTrue(pwl_path.is_file())
            pairs = read_pwl_data(pwl_path)
            self.assertEqual(pairs[-1][1], 0.0)
            self.assertAlmostEqual(pairs[-1][0], 4 / 8_000)
            audio_values = [value for _time, value in pairs[:-1]]
            self.assertAlmostEqual(
                max(audio_values) - min(audio_values), 0.010, places=12
            )
            self.assertAlmostEqual(
                math.fsum(audio_values) / len(audio_values), 0.0, places=12
            )
            self.assertTrue((output_root / "pwl_manifest.csv").is_file())


class TransientTests(unittest.TestCase):
    """Verify source replacement, Vpp scaling, and analog raw parsing."""

    def test_vsin_or_v4_name_is_discovered_and_replaced(self) -> None:
        """Transient generation must preserve source terminals but remove SIN."""

        template = (
            ".title transient test\n"
            "Vbias bias 0 DC 0.9\n"
            "Vsin /vin bias DC 0 SIN(0 1 1k) AC 1\n"
            "Rload /vin 0 1k\n"
            ".end\n"
        )
        self.assertEqual(
            find_input_voltage_source(template),
            ("Vsin", "/vin", "bias"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = make_transient_netlist(
                template,
                root / "params.inc",
                root / "stimulus.pwl.inc",
                TransientSettings(),
                1.2,
            )
        self.assertNotIn("SIN(", text)
        self.assertNotIn("Vsin /vin bias DC", text)
        self.assertIn("stimulus.pwl.inc", text)
        self.assertIn(".tran 1e-05 1.2 0 5e-06", text)

    def test_pwl_include_uses_original_source_and_final_zero(self) -> None:
        """The generated include must be a legal continued PWL source line."""

        text = make_pwl_source_include(
            "Vsin",
            "/vin",
            "bias",
            [(0.0, -0.001), (0.001, 0.002), (0.002, 0.0)],
        )
        self.assertIn("Vsin /vin bias PWL(0 -0.001", text)
        self.assertIn("+ 0.002 0)", text)

    def test_transient_raw_contains_only_four_requested_analog_voltages(
        self,
    ) -> None:
        """The viewer reads Vin, Vfilt, Venv, and Vref without Vcomp logic."""

        raw_text = (
            "Title: transient fixture\n"
            "Plotname: Transient Analysis\n"
            "Flags: real\n"
            "No. Variables: 6\n"
            "No. Points: 4\n"
            "Variables:\n"
            "\t0\ttime\ttime\n"
            "\t1\tv(/vin)\tvoltage\n"
            "\t2\tv(/v_filt_out)\tvoltage\n"
            "\t3\tv(/v_detect_out)\tvoltage\n"
            "\t4\tv(/v_ref)\tvoltage\n"
            "\t5\tv(/v_comp_out)\tvoltage\n"
            "Values:\n"
            "0\t0\n\t0.9\n\t0.9\n\t0.90\n\t0.91\n\t0\n"
            "1\t0.001\n\t0.901\n\t0.9005\n\t0.92\n\t0.91\n\t1.8\n"
            "2\t0.002\n\t0.899\n\t0.8995\n\t0.93\n\t0.91\n\t1.8\n"
            "3\t0.003\n\t0.9\n\t0.9\n\t0.90\n\t0.91\n\t0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tran.raw"
            path.write_text(raw_text, encoding="utf-8")
            rows = read_transient_raw(path, TransientSettings())
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(rows[0]), 5)

    def test_transient_input_pwl_scales_to_requested_vpp(self) -> None:
        """A 10 mVpp file can be reused at another Vpp without file mutation."""

        pairs = [(0.0, -0.004), (0.001, 0.006), (0.002, 0.0)]
        scaled = scale_pwl_pairs_to_vpp(pairs, 0.025)
        voltages = [voltage for _time, voltage in scaled]
        self.assertAlmostEqual(max(voltages) - min(voltages), 0.025)
        self.assertEqual(scaled[-1][1], 0.0)
        self.assertEqual(pairs[0][1], -0.004)

    @unittest.skipIf(os.name == "nt", "POSIX shebang test double")
    def test_transient_runner_writes_analog_csv_without_colored_interval_data(
        self,
    ) -> None:
        """The runner must omit the removed Venv>Vref post-processing files."""

        fake = Path(__file__).with_name("fake_ngspice.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pwl = root / "go" / "sample.pwl"
            pwl.parent.mkdir()
            pwl.write_text(
                "0 -0.005\n0.0005 0.005\n0.001 0\n",
                encoding="ascii",
            )
            simulator = TransientSimulator(log=lambda _text: None)
            results = simulator.run(
                [load_channels(DEFAULT_TABLE)[0]],
                [pwl],
                Path(__file__).parents[1] / "netlist_template.cir",
                root / "results",
                str(fake),
                TransientSettings(stop_time_s=0.003, input_vpp_v=0.020),
            )
            self.assertEqual(len(results), 1)
            run_dir = results[0].run_dir
            self.assertTrue((run_dir / "transient.csv").is_file())
            self.assertFalse(
                (run_dir / "venv_above_vref_intervals.csv").exists()
            )
            csv_text = (run_dir / "transient.csv").read_text(encoding="utf-8")
            self.assertEqual(
                csv_text.splitlines()[0],
                "time_s,vin_v,vfilt_out_v,venv_out_v,vref_v",
            )
            include_text = (
                run_dir.parent / "stimulus.pwl.inc"
            ).read_text(encoding="ascii")
            self.assertIn("PWL(0 -0.01", include_text)
            self.assertIn("+ 0.0005 0.01", include_text)
            self.assertIn("0 -0.005", pwl.read_text(encoding="ascii"))
            netlist = (run_dir / "channel_00_tran.cir").read_text(
                encoding="utf-8"
            )
            self.assertIn(".tran 1e-05 0.003 0 5e-06", netlist)
            self.assertNotIn(" SIN(", netlist)


class TransientDisplayTests(unittest.TestCase):
    """Verify human-readable settings and the ms/mV result-view contract."""

    def test_time_parser_accepts_scientific_and_spice_suffixes(self) -> None:
        """The old 1e-5 and clearer 10u forms must be exactly equivalent."""

        self.assertAlmostEqual(parse_time_seconds("1e-5"), 10e-6)
        self.assertAlmostEqual(parse_time_seconds("10u"), 10e-6)
        self.assertAlmostEqual(parse_time_seconds("5us"), 5e-6)
        self.assertAlmostEqual(parse_time_seconds("1.5ms"), 1.5e-3)

    def test_transient_settings_include_runtime_input_vpp(self) -> None:
        """The requested PWL amplitude belongs to transient run settings."""

        settings = TransientSettings()
        names = {field.name for field in fields(settings)}
        self.assertEqual(
            names,
            {
                "output_step_s",
                "stop_time_s",
                "maximum_step_s",
                "input_vpp_v",
                "vin_node",
                "vfilt_node",
                "venv_node",
                "vref_node",
                "pspice_compat",
            },
        )

    def test_default_max_step_is_conservative_for_eight_kilohertz(self) -> None:
        """Five microseconds is below the 20-points/cycle 8 kHz bound."""

        bound = recommended_maximum_step_s(8000.0, points_per_cycle=20)
        self.assertAlmostEqual(bound, 6.25e-6)
        self.assertLessEqual(TransientSettings().maximum_step_s, bound)

    def test_plot_rows_convert_seconds_volts_to_ms_mv(self) -> None:
        """Only the display copy changes units; raw result rows stay in SI."""

        rows = [
            (0.001, 0.900, 0.901, 0.920, 0.915),
            (0.002, 0.905, 0.899, 0.930, 0.915),
        ]
        times, vin, vfilt, venv, vref = transient_rows_to_ms_mv(rows)
        self.assertEqual(times, [1.0, 2.0])
        self.assertEqual(vin, [900.0, 905.0])
        self.assertEqual(vfilt, [901.0, 899.0])
        self.assertEqual(venv, [920.0, 930.0])
        self.assertEqual(vref, [915.0, 915.0])
        self.assertEqual(rows[0][0], 0.001)
        self.assertEqual(rows[0][1], 0.900)

    def test_auto_y_range_uses_ninety_percent_of_axis_height(self) -> None:
        """Data min/max must occupy exactly 90% of each automatic Y range."""

        values = [899.75, 900.0, 900.25]
        low, high = data_occupancy_limits(values)
        self.assertAlmostEqual(
            (max(values) - min(values)) / (high - low),
            0.90,
            places=12,
        )
        ac_low, ac_high = data_occupancy_limits([-12.0, 3.0])
        self.assertAlmostEqual(15.0 / (ac_high - ac_low), 0.90, places=12)

    def test_transient_renderer_uses_numeric_ranges_without_span_selector(self) -> None:
        """UI source must retain units and remove slow drag-range selection."""

        render_source = inspect.getsource(SweeperGUI._render_transient_result)
        controls_source = inspect.getsource(
            SweeperGUI._build_transient_range_controls
        )
        embed_source = inspect.getsource(SweeperGUI._embed_multi_axes_figure)
        self.assertIn("Time [ms]", render_source)
        self.assertIn("Vin [mV]", render_source)
        self.assertIn('text="X [ms]"', controls_source)
        self.assertIn('text="Y [mV]"', controls_source)
        self.assertIn("row=0", controls_source)
        self.assertIn("row=1", controls_source)
        self.assertNotIn("row=2", controls_source)
        self.assertNotIn("Venv_out > Vref", render_source)
        self.assertNotIn("axvspan", render_source)
        self.assertNotIn("X 구간 드래그", render_source)
        self.assertNotIn("SpanSelector", render_source)
        self.assertNotIn("NavigationToolbar2Tk", embed_source)

    def test_manual_transient_ranges_are_saved_and_reapplied(self) -> None:
        """Channel/result rerenders reuse independent manual X and per-graph Y."""

        x_apply_source = inspect.getsource(
            SweeperGUI._apply_transient_x_range
        )
        y_apply_source = inspect.getsource(
            SweeperGUI._apply_transient_y_range
        )
        render_source = inspect.getsource(SweeperGUI._render_transient_result)
        self.assertIn(
            "transient_manual_x_limits_ms = x_limits",
            x_apply_source,
        )
        self.assertIn(
            "transient_manual_y_limits_mv[target] = y_limits",
            y_apply_source,
        )
        self.assertIn('transient_x_mode_var.get() == "수동"', render_source)
        self.assertIn('transient_y_modes.get(name) == "수동"', render_source)
        self.assertIn("self.transient_manual_x_limits_ms", render_source)
        self.assertIn("self.transient_manual_y_limits_mv[name]", render_source)

    def test_transient_result_selector_splits_pwl_and_sixteen_channels(
        self,
    ) -> None:
        """PWL stays a dropdown while only completed channel options are enabled."""

        build = inspect.getsource(SweeperGUI._build_transient_tab)
        install = inspect.getsource(SweeperGUI._install_transient_results)
        enable = inspect.getsource(
            SweeperGUI._refresh_transient_result_channel_buttons
        )
        self.assertIn('text="결과 PWL"', build)
        self.assertIn("for channel_number in range(16)", build)
        self.assertIn('state="disabled"', build)
        self.assertIn(
            "lookup[(label, result.channel.ch)] = result",
            install,
        )
        self.assertIn(
            '"normal" if channel_number in available else "disabled"',
            enable,
        )

    def test_result_selector_enables_only_completed_channels_dynamically(
        self,
    ) -> None:
        """Installing ch00/ch02 results must leave every other option disabled."""

        channels = load_channels(DEFAULT_TABLE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stimulus = root / "down" / "sample.pwl"
            results = [
                TransientResult(
                    channel=channels[channel_number],
                    stimulus_path=stimulus,
                    run_dir=root / f"ch{channel_number:02d}",
                    rows=[],
                    log_text="",
                )
                for channel_number in (0, 2)
            ]
            gui = SweeperGUI.__new__(SweeperGUI)
            gui.transient_folder_var = DummyVar(str(root))
            gui.transient_result_var = DummyVar("")
            gui.transient_result_channel_var = DummyVar(-1)
            gui.transient_result_combo = DummyWidget()
            gui.transient_result_channel_buttons = {
                channel_number: DummyWidget()
                for channel_number in range(16)
            }
            gui.transient_result_lookup = {}
            gui.transient_result_stimulus_paths = {}
            rendered: list[TransientResult] = []
            gui._render_transient_result = rendered.append
            gui._refresh_transient_component_state = lambda: None

            gui._install_transient_results(results)
            self.assertEqual(
                gui.transient_result_combo.options["values"],
                ("down/sample.pwl",),
            )
            enabled = {
                channel_number
                for channel_number, button
                in gui.transient_result_channel_buttons.items()
                if button.options["state"] == "normal"
            }
            self.assertEqual(enabled, {0, 2})
            self.assertEqual(gui.transient_result_channel_var.get(), 0)
            self.assertEqual(rendered[-1].channel.ch, 0)

            gui.transient_result_channel_var.set(2)
            gui._on_transient_result_channel_changed()
            self.assertEqual(rendered[-1].channel.ch, 2)


class UiInvariantTests(unittest.TestCase):
    """Lock the requested tab order and simplified V2 display text."""

    def test_four_workflow_tabs_have_requested_order_and_dc_default(self) -> None:
        """DC, AC, transient, and PWL must be separate top-level tabs."""

        source = inspect.getsource(SweeperGUI._build)
        self.assertLess(
            source.index('self.workflow_notebook.add(self.dc_page'),
            source.index('self.workflow_notebook.add(self.ac_page'),
        )
        self.assertLess(
            source.index('self.workflow_notebook.add(self.ac_page'),
            source.index('self.workflow_notebook.add(self.transient_page'),
        )
        self.assertLess(
            source.index('self.workflow_notebook.add(self.transient_page'),
            source.index('self.workflow_notebook.add(self.converter_page'),
        )
        self.assertIn("self.workflow_notebook.select(self.dc_page)", source)

    def test_simulation_completion_does_not_switch_visible_result_tabs(
        self,
    ) -> None:
        """Worker completion may redraw data but must not select another tab."""

        poll_source = inspect.getsource(SweeperGUI._poll_events)
        transient_render = inspect.getsource(
            SweeperGUI._render_transient_result
        )
        results_source = inspect.getsource(SweeperGUI._render_results)
        self.assertNotIn("workflow_notebook.select", poll_source)
        self.assertNotIn("transient_notebook.select", poll_source)
        self.assertNotIn("transient_notebook.select", transient_render)
        self.assertIn("selected_ac_tab", results_source)

    def test_component_apply_starts_dc_operating_point(self) -> None:
        """The editor apply button must launch OP after committing values."""

        source = inspect.getsource(SweeperGUI._apply_component_editor_from_gui)
        self.assertIn('("op",)', source)
        self.assertIn("selected_channels=(updated,)", source)

    def test_ac_and_transient_commit_visible_component_edits(self) -> None:
        """Both non-DC run buttons must apply the same in-memory editor values."""

        ac_source = inspect.getsource(SweeperGUI._start_run)
        transient_source = inspect.getsource(SweeperGUI._start_transient_run)
        self.assertIn("self._commit_component_editor()", ac_source)
        self.assertIn("self._commit_component_editor()", transient_source)

    def test_ac_manual_range_variables_are_not_reset_by_result_refresh(
        self,
    ) -> None:
        """Only the explicit auto-range button may clear AC manual bounds."""

        render_source = inspect.getsource(SweeperGUI._render_results)
        reset_source = inspect.getsource(
            SweeperGUI._reset_magnitude_axis_limits
        )
        self.assertNotIn("mag_x_min_var.set", render_source)
        self.assertNotIn("mag_y_min_var.set", render_source)
        self.assertIn("variable.set(\"\")", reset_source)

    def test_cursor_hover_is_connected_but_guarded_by_button_arm(self) -> None:
        """Motion follows a line only after the explicit add-cursor button."""

        cursor_source = inspect.getsource(InteractivePlotCursor)
        motion_source = inspect.getsource(InteractivePlotCursor._on_motion)
        embed_source = inspect.getsource(SweeperGUI._embed_figure)
        transient_embed = inspect.getsource(
            SweeperGUI._embed_multi_axes_figure
        )
        self.assertIn("motion_notify_event", cursor_source)
        self.assertIn("if not self.armed", motion_source)
        self.assertIn("preview_vertical", cursor_source)
        self.assertIn("supports_blit", cursor_source)
        self.assertIn('text="커서 추가"', embed_source)
        self.assertIn('text="커서 추가"', transient_embed)
        self.assertIn("self.artists.append", cursor_source)
        self.assertIn("cursor.dispose()", inspect.getsource(
            SweeperGUI._dispose_cursor_group
        ))

    def test_margin_editor_replaces_direct_vref_editor(self) -> None:
        """Vmargin is editable, while calculated Vref is display-only."""

        source = inspect.getsource(SweeperGUI._build_op_detail_tables)
        self.assertIn("목표 V_margin [mV]", source)
        self.assertIn("계산 Vref [mV]", source)
        self.assertIn('state="readonly"', source)
        self.assertNotIn("R7/R8 계산 Vref [mV]", source)
        self.assertIn("Vmargin = Vref − Vdet", source)
        self.assertIn("Vdet + 목표 Vmargin", source)
        self.assertIn("합계 고정 안 함", source)
        self.assertIn("format_resistance_kohm(value)", source)

    def test_dc_refresh_preserves_detail_tab_pane_and_scroll(self) -> None:
        """Replacing OP widgets must restore the user's visible layout state."""

        capture = inspect.getsource(
            SweeperGUI._capture_operating_point_view_state
        )
        restore = inspect.getsource(
            SweeperGUI._restore_operating_point_view_state
        )
        render = inspect.getsource(SweeperGUI._render_results)
        op_render = inspect.getsource(SweeperGUI._render_operating_point)
        detail = inspect.getsource(SweeperGUI._build_op_detail_tables)
        init = inspect.getsource(SweeperGUI.__init__)
        self.assertIn("op_detail_selected_tab", capture)
        self.assertIn("sashpos(0)", capture)
        self.assertIn("canvas.xview()[0]", capture)
        self.assertIn("sashpos(0,", restore)
        self.assertIn("notebook.select(selected_detail)", restore)
        self.assertIn("_capture_operating_point_view_state", render)
        self.assertIn("notebook.select(selected_detail)", detail)
        self.assertIn("OP_DEFAULT_SCHEMATIC_FRACTION", init)
        self.assertIn("width=330", op_render)
        self.assertNotIn("root.geometry", op_render)

    def test_dc_apply_freezes_view_state_before_async_rerun(self) -> None:
        """The apply click preserves detail tab, sash, and scroll across both OP runs."""

        apply_source = inspect.getsource(
            SweeperGUI._apply_component_editor_from_gui
        )
        event_source = inspect.getsource(
            SweeperGUI._on_op_detail_tab_changed
        )
        render_source = inspect.getsource(SweeperGUI._render_results)
        self.assertIn("_capture_operating_point_view_state", apply_source)
        self.assertIn("op_pending_view_state", apply_source)
        self.assertIn("event_notebook", event_source)
        self.assertIn("pending_view", render_source)

    def test_transient_tab_reports_changed_or_stale_component_values(
        self,
    ) -> None:
        """DC edits are visible in Transient before and after a matching rerun."""

        build = inspect.getsource(SweeperGUI._build_transient_tab)
        refresh = inspect.getsource(
            SweeperGUI._refresh_transient_component_state
        )
        commit = inspect.getsource(SweeperGUI._commit_component_editor)
        self.assertIn("transient_component_state_var", build)
        self.assertIn("기본 CSV 대비 변경됨", refresh)
        self.assertIn("아직 적용하지 않은 변경", refresh)
        self.assertIn("재실행 필요", refresh)
        self.assertIn("현재 소자값 반영됨", refresh)
        self.assertIn("_refresh_transient_component_state", commit)

    def test_component_state_message_distinguishes_stale_and_current_result(
        self,
    ) -> None:
        """A DC edit marks an old Transient result stale until rerun."""

        original = load_channels(DEFAULT_TABLE)[0]
        changed = apply_component_updates(
            original,
            {
                "RA": "13.00",
                "R1": format_resistance_kohm(original.r1_kohm),
                "R2": format_resistance_kohm(original.r2_kohm),
                "R4": format_resistance_kohm(original.r4_kohm),
                "R5": format_resistance_kohm(original.r5_kohm),
                "R6": format_resistance_kohm(original.r6_kohm),
                "R7": format_resistance_kohm(original.r7_kohm),
                "R8": format_resistance_kohm(original.r8_kohm),
                "C1": str(original.c_nf),
                "C3": str(original.c3_nf),
            },
        )
        gui = SweeperGUI.__new__(SweeperGUI)
        gui.default_channels = [original]
        gui.channels = [changed]
        gui.transient_component_state_var = DummyVar()
        gui.transient_result_var = DummyVar("sample.pwl")
        gui.transient_result_channel_var = DummyVar(0)
        old_result = TransientResult(
            channel=original,
            stimulus_path=Path("sample.pwl"),
            run_dir=Path("."),
            rows=[],
            log_text="",
        )
        gui.transient_result_lookup = {("sample.pwl", 0): old_result}
        gui._refresh_transient_component_state()
        self.assertIn(
            "변경 전 값",
            str(gui.transient_component_state_var.get()),
        )

        new_result = TransientResult(
            channel=changed,
            stimulus_path=Path("sample.pwl"),
            run_dir=Path("."),
            rows=[],
            log_text="",
        )
        gui.transient_result_lookup = {("sample.pwl", 0): new_result}
        gui._refresh_transient_component_state()
        self.assertIn(
            "현재 소자값 반영됨",
            str(gui.transient_component_state_var.get()),
        )

    def test_returning_to_transient_redraws_without_resetting_result(self) -> None:
        """A hidden transient canvas is repainted without rebuilding its data."""

        source = inspect.getsource(SweeperGUI._on_workflow_tab_changed)
        self.assertIn("canvas.draw_idle()", source)
        self.assertNotIn("_render_transient_result", source)
        self.assertNotIn("_reset_transient_axis_ranges", source)

    def test_v2_arrow_label_is_removed(self) -> None:
        """The GUI must not describe V2 with the confusing rail arrow."""

        source = inspect.getsource(SweeperGUI)
        self.assertNotIn("0.9 V → 1.8 V", source)

    def test_removed_compatibility_shims_do_not_remain_in_runtime(self) -> None:
        """Dead v9/v10 aliases and no-op range helpers must stay removed."""

        module_source = Path(inspect.getsourcefile(SweeperGUI) or "").read_text(
            encoding="utf-8"
        )
        for dead_name in (
            "DIVIDER_TOTAL_KOHM",
            "component_threshold_var",
            "centered_transient_limits_mv",
            "_apply_transient_axis_ranges",
            "_reset_transient_axis_ranges",
            "_push_transient_view_history",
            "select_result_tab",
        ):
            self.assertNotIn(dead_name, module_source)


class DocumentationTests(unittest.TestCase):
    """Keep function-level documentation synchronized with the shipped code."""

    def test_every_function_and_class_has_a_docstring(self) -> None:
        """Production and test Python definitions must all explain their role."""

        project_root = Path(__file__).parents[1]
        paths = (
            project_root / "ngspice_channel_sweeper.py",
            project_root / "ngspice_runner.py",
            project_root / "wav_pwl.py",
            project_root / "tests" / "test_core.py",
            project_root / "tests" / "fake_ngspice.py",
        )
        missing: list[str] = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ) and ast.get_docstring(node) is None:
                    missing.append(f"{path.name}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
