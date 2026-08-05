"""Independent Active Detector controls, component editing, and plotting."""

from __future__ import annotations

import threading
from pathlib import Path

from ..models import Channel, DetectorResult, DetectorSettings
from ..results import detector_ac_comparison_rows
from ..simulation import DetectorSimulator, find_ngspice
from ..values import data_occupancy_limits
from .base import AppController, configure_plot_fonts
from .cursors import InteractivePlotCursor


class DetectorTabController(AppController):
    """Own Active Detector state without mutating PWL transient workflow state."""

    def _build_detector_tab(self) -> None:
        """Build gated-sine settings, shared components, plot, and result log."""

        outer = self.ttk.Frame(self.detector_page, padding=10)
        outer.pack(fill="both", expand=True)
        settings_row = self.ttk.Frame(outer)
        settings_row.pack(fill="x")
        settings_row.columnconfigure(0, weight=4)
        settings_row.columnconfigure(1, weight=3)
        settings_row.columnconfigure(2, weight=3)

        stimulus = self.ttk.LabelFrame(
            settings_row,
            text="Gated sine 설정",
            padding=7,
        )
        stimulus.grid(row=0, column=0, sticky="nsew")
        fields = (
            ("정현파 주파수 [Hz]", self.detector_frequency_var),
            ("입력 진폭", self.detector_amplitude_var),
            ("Gate ON [ms]", self.detector_gate_on_var),
            ("Gate 지속시간 [ms]", self.detector_gate_duration_var),
            ("전체 해석시간 [ms]", self.detector_total_time_var),
            ("maximum timestep [us]", self.detector_max_step_var),
        )
        for row, (label, variable) in enumerate(fields):
            self.ttk.Label(stimulus, text=label).grid(
                row=row,
                column=0,
                sticky="e",
                padx=(0, 5),
                pady=2,
            )
            self.ttk.Entry(stimulus, textvariable=variable, width=15).grid(
                row=row,
                column=1,
                sticky="ew",
                pady=2,
            )
        amplitude_unit = self.ttk.Combobox(
            stimulus,
            textvariable=self.detector_amplitude_unit_var,
            values=("mVpp", "Vpp"),
            state="readonly",
            width=7,
        )
        amplitude_unit.grid(row=1, column=2, sticky="w", padx=(4, 0), pady=2)
        stimulus.columnconfigure(1, weight=1)
        self.ttk.Checkbutton(
            stimulus,
            text="detector.csv 생성 (detector.raw와 중복, 대용량)",
            variable=self.detector_write_csv_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(5, 0))
        self.ttk.Label(
            stimulus,
            textvariable=self.detector_result_state_var,
            foreground="#9a5500",
            justify="left",
            wraplength=430,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))

        components = self._build_shared_component_editor(
            settings_row,
            "detector",
            note=(
                "R4 또는 R5 변경 시 R6 = (R4 * R5) / (R4 + R5).\n"
                "Vthr를 입력하면 R7/R8이 다시 계산됩니다.\n"
                "U3 opamp만 교체되고 필터부 U1, U2는 OPA379 고정입니다."
            ),
        )
        components.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        channels_frame = self._build_channel_selector(
            settings_row,
            "detector_channel_list",
            self._select_all_detector_channels,
            self._clear_detector_channels,
            title="Detector 채널",
            on_select=self._on_detector_channel_changed,
        )
        channels_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=(7, 6))
        self.detector_run_button = self.ttk.Button(
            action,
            text="Active Detector 실행",
            command=self._start_detector_run,
        )
        self.detector_run_button.pack(side="left")
        self.detector_stop_button = self.ttk.Button(
            action,
            text="중지",
            command=self._stop_detector_run,
            state="disabled",
        )
        self.detector_stop_button.pack(side="left", padx=(5, 0))
        self.detector_progress = self.ttk.Progressbar(
            action,
            mode="indeterminate",
            length=170,
        )
        self.detector_progress.pack(side="left", padx=10)
        self.ttk.Label(action, textvariable=self.detector_status_var).pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.detector_notebook = self.ttk.Notebook(outer)
        self.detector_notebook.pack(fill="both", expand=True)
        self.detector_plot_frame = self.ttk.Frame(self.detector_notebook)
        self.detector_log_frame = self.ttk.Frame(self.detector_notebook)
        self.detector_notebook.add(
            self.detector_plot_frame,
            text="Active Detector 결과",
        )
        self.detector_notebook.add(
            self.detector_log_frame,
            text="Active Detector 로그",
        )
        self.ttk.Label(
            self.detector_plot_frame,
            text="Active Detector 실행 후 네 파형과 측정 결과가 표시됩니다.",
        ).pack(expand=True)
        self.detector_log_text = self.tk.Text(
            self.detector_log_frame,
            wrap="none",
            height=12,
        )
        log_y = self.ttk.Scrollbar(
            self.detector_log_frame,
            orient="vertical",
            command=self.detector_log_text.yview,
        )
        log_x = self.ttk.Scrollbar(
            self.detector_log_frame,
            orient="horizontal",
            command=self.detector_log_text.xview,
        )
        self.detector_log_text.configure(
            yscrollcommand=log_y.set,
            xscrollcommand=log_x.set,
        )
        self.detector_log_text.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")
        self.detector_log_frame.rowconfigure(0, weight=1)
        self.detector_log_frame.columnconfigure(0, weight=1)
        self._update_detector_run_button()

    def _populate_detector_channels(self) -> None:
        """Mirror the current component-version channels into Detector controls."""

        if not hasattr(self, "detector_channel_list"):
            return
        previous = {
            self.channels[index].ch
            for index in self.detector_channel_list.curselection()
            if 0 <= index < len(self.channels)
        }
        self.detector_channel_list.delete(0, "end")
        restored = False
        for index, channel in enumerate(self.channels):
            self.detector_channel_list.insert(
                "end", f"ch{channel.ch:02d}   fc={channel.f_c_hz:g} Hz"
            )
            if channel.ch in previous:
                self.detector_channel_list.selection_set(index)
                restored = True
        if self.channels and not restored:
            self.detector_channel_list.selection_set(0)
        self._on_detector_channel_changed()

    def _selected_detector_channels(self) -> list[Channel]:
        """Return every channel ticked in the Detector list."""

        if not hasattr(self, "detector_channel_list"):
            return []
        return [
            self.channels[index]
            for index in self.detector_channel_list.curselection()
            if 0 <= index < len(self.channels)
        ]

    def _detector_channel_number(self) -> int:
        """Return the channel whose shared component values are being edited.

        The editor shows one channel at a time, so the loaded channel wins over
        the list order even when several channels are queued for execution.
        """

        if self.detector_selected_channel is not None:
            return self.detector_selected_channel
        selected = self._selected_detector_channels()
        if not selected:
            raise ValueError("Active Detector 채널을 선택하세요.")
        return selected[0].ch

    def _select_all_detector_channels(self) -> None:
        """Select every loaded channel for Active Detector."""

        self.detector_channel_list.selection_set(0, "end")
        self._on_detector_channel_changed()

    def _clear_detector_channels(self) -> None:
        """Clear the Detector channel selection."""

        self.detector_channel_list.selection_clear(0, "end")
        self._update_detector_run_button()

    def _on_detector_channel_changed(self, _event: object = None) -> None:
        """Load the first selected channel's shared values and cached result."""

        from tkinter import messagebox

        selected = self._selected_detector_channels()
        if not selected:
            self.detector_selected_channel = None
            self._update_detector_run_button()
            return
        first = selected[0].ch
        if (
            first == self.detector_selected_channel
            and self._shared_component_channel("detector") == first
        ):
            # Both the result view and the shared panel already show this
            # channel; skipping early would otherwise leave a stale panel.
            self._update_detector_run_button()
            return
        try:
            self.detector_selected_channel = first
            self._load_shared_component_fields("detector", first)
            result = self.detector_result_cache.get(first)
            if result is not None:
                self.last_detector_result = result
                self._install_detector_result(result)
            else:
                self.last_detector_result = None
                self._clear_frame(self.detector_plot_frame)
                self.ttk.Label(
                    self.detector_plot_frame,
                    text=f"ch{first:02d} Active Detector 결과가 없습니다.",
                ).pack(expand=True)
            self._refresh_detector_component_state()
            self._update_detector_run_button()
        except Exception as exc:
            messagebox.showerror("Detector 소자값 오류", str(exc))






    def _refresh_detector_component_state(self) -> None:
        """Mark cached Detector results current or stale by component revision."""

        if not hasattr(self, "detector_result_state_var"):
            return
        channel_number = self.detector_selected_channel
        result = (
            self.detector_result_cache.get(channel_number)
            if channel_number is not None
            else None
        )
        if result is None:
            self.detector_result_state_var.set(
                f"현재 소자값 revision {self.component_revision}; 결과 없음"
            )
        elif result.is_stale(self.component_revision):
            self.detector_result_state_var.set(
                "소자 변경 전 결과, 재실행 필요"
            )
        else:
            self.detector_result_state_var.set(
                f"현재 소자값 결과; revision {result.component_revision}"
            )

    def _detector_settings(self) -> DetectorSettings:
        """Parse editable GUI units into an immutable SI-unit settings object."""

        try:
            amplitude = float(self.detector_amplitude_var.get())
            input_vpp_v = (
                amplitude / 1000.0
                if self.detector_amplitude_unit_var.get() == "mVpp"
                else amplitude
            )
            settings = DetectorSettings(
                frequency_hz=float(self.detector_frequency_var.get()),
                input_vpp_v=input_vpp_v,
                gate_on_s=float(self.detector_gate_on_var.get()) / 1000.0,
                gate_duration_s=(
                    float(self.detector_gate_duration_var.get()) / 1000.0
                ),
                total_time_s=float(self.detector_total_time_var.get()) / 1000.0,
                maximum_step_s=float(self.detector_max_step_var.get()) / 1e6,
                pspice_compat=self.pspice_compat_var.get(),
                write_csv=bool(self.detector_write_csv_var.get()),
            )
        except ValueError as exc:
            raise ValueError("Active Detector 설정에는 숫자를 입력하세요.") from exc
        settings.validate()
        return settings

    def _update_detector_run_button(self) -> None:
        """Disable Detector execution during any simulation worker."""

        if not hasattr(self, "detector_run_button"):
            return
        running = bool(
            getattr(self, "detector_worker", None)
            and self.detector_worker.is_alive()
        ) or bool(self.worker and self.worker.is_alive()) or bool(
            self.transient_worker and self.transient_worker.is_alive()
        )
        count = len(self._selected_detector_channels())
        self.detector_run_button.configure(
            text=(
                f"Active Detector 실행 ({count} jobs)"
                if count
                else "Detector 채널 선택 필요"
            ),
            state="disabled" if running or count == 0 else "normal",
        )

    def _start_detector_run(self) -> None:
        """Launch a Detector worker using a component/revision snapshot."""

        from tkinter import messagebox

        if (
            self.detector_worker is not None
            and self.detector_worker.is_alive()
        ):
            return
        try:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("AC/OP 실행이 끝난 뒤 Detector를 시작하세요.")
            if self.transient_worker and self.transient_worker.is_alive():
                raise RuntimeError("Transient 실행이 끝난 뒤 Detector를 시작하세요.")
            self._commit_shared_components("detector")
            selected = self._selected_detector_channels()
            if not selected:
                raise ValueError("Active Detector 채널을 하나 이상 선택하세요.")
            settings = self._detector_settings()
            warning = settings.timestep_warning()
            if warning and not messagebox.askyesno(
                "Detector timestep 경고",
                warning + "\n\n현재 값으로 계속 실행할까요?",
            ):
                return
            ngspice = find_ngspice(self.ngspice_var.get())
            netlist = Path(self.netlist_var.get()).expanduser().resolve()
            output = Path(self.output_var.get()).expanduser().resolve()
            if not netlist.is_file():
                raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {netlist}")
            output.mkdir(parents=True, exist_ok=True)
            revision = self.component_revision
        except Exception as exc:
            messagebox.showerror("Active Detector 설정 오류", str(exc))
            return
        self.detector_log_text.delete("1.0", "end")
        self.detector_simulator = DetectorSimulator(
            lambda text: self.events.put(("detector_log", text))
        )
        self.detector_run_button.configure(state="disabled")
        self.detector_stop_button.configure(state="normal")
        self.op_run_button.configure(state="disabled")
        self.ac_run_button.configure(state="disabled")
        self.transient_run_button.configure(state="disabled")
        self.detector_channel_list.configure(state="disabled")
        self.detector_progress.start(12)
        # A new run supersedes the previous cache for the selected channels.
        self.detector_live_rendered = False
        self.detector_completed_jobs = 0
        self.detector_total_jobs = len(selected)
        self.detector_status_var.set(
            f"Active Detector {len(selected)} jobs 실행 중..."
        )

        def worker() -> None:
            """Run the blocking detector jobs outside Tk's event loop."""

            try:
                results = self.detector_simulator.run_channels(
                    selected,
                    netlist,
                    output,
                    ngspice,
                    settings,
                    revision,
                    on_result=lambda result: self.events.put(
                        ("detector_result", result)
                    ),
                )
                self.events.put(("detector_done", results))
            except Exception as exc:
                self.events.put(("detector_error", exc))

        self.detector_worker = threading.Thread(target=worker, daemon=True)
        self.detector_worker.start()

    def _stop_detector_run(self) -> None:
        """Forward a stop request only to the Active Detector simulator."""

        if self.detector_simulator:
            self.detector_simulator.stop()
            self.detector_status_var.set("Active Detector 중지 요청 중...")

    def _finish_detector_controls(self) -> None:
        """Restore all run controls after Detector success or failure."""

        self.detector_progress.stop()
        self.detector_worker = None
        self.detector_stop_button.configure(state="disabled")
        self.detector_channel_list.configure(state="normal")
        self._update_run_button()
        self._update_transient_run_button()
        self._update_detector_run_button()

    def _install_detector_result(self, result: DetectorResult) -> None:
        """Publish one finished channel without disturbing the current view.

        Only the first result of a run is drawn automatically.  Later channels
        just land in the cache so an in-progress run never replaces the graph,
        cursors, or ranges the user is inspecting.
        """

        self.detector_result_cache[result.channel.ch] = result
        if not self.detector_live_rendered:
            self.detector_live_rendered = True
            self.detector_selected_channel = result.channel.ch
            self.last_detector_result = result
            self._refresh_detector_component_state()
            self._render_detector_result(result)
            return
        if result.channel.ch == self.detector_selected_channel:
            self.last_detector_result = result
            self._refresh_detector_component_state()
            self._render_detector_result(result)

    def _store_detector_axis_ranges(self, mode: str | None = None) -> None:
        """Save independent absolute/AC comparison manual range text."""

        active = mode or self.detector_mode_var.get()
        if active not in {"절대전압", "AC 비교"}:
            return
        self.detector_manual_x_limits_ms[active] = (
            self.detector_x_min_var.get(),
            self.detector_x_max_var.get(),
        )
        self.detector_manual_y_limits_mv[active] = (
            self.detector_y_min_var.get(),
            self.detector_y_max_var.get(),
        )

    def _restore_detector_axis_ranges(self, mode: str) -> None:
        """Load manual range text for one detector display mode."""

        x_min, x_max = self.detector_manual_x_limits_ms.get(mode, ("", ""))
        y_min, y_max = self.detector_manual_y_limits_mv.get(mode, ("", ""))
        self.detector_x_min_var.set(x_min)
        self.detector_x_max_var.set(x_max)
        self.detector_y_min_var.set(y_min)
        self.detector_y_max_var.set(y_max)

    def _detector_axis_limits(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Validate optional Detector limits; both axes remain ordinary linear."""

        x_min, x_max = self._parse_axis_limits(
            self.detector_x_min_var.get(),
            self.detector_x_max_var.get(),
            "Detector X축",
            False,
        )
        y_min, y_max = self._parse_axis_limits(
            self.detector_y_min_var.get(),
            self.detector_y_max_var.get(),
            "Detector Y축",
            False,
        )
        return x_min, x_max, y_min, y_max

    def _on_detector_mode_changed(self, _event: object = None) -> None:
        """Switch display data and restore the selected mode's manual ranges."""

        previous = self.detector_previous_mode
        self._store_detector_axis_ranges(previous)
        current = self.detector_mode_var.get()
        if current not in {"절대전압", "AC 비교"}:
            self.detector_mode_var.set(previous)
            return
        self.detector_display_mode = current
        self._restore_detector_axis_ranges(current)
        self.detector_previous_mode = current
        self._refresh_detector_plot()

    def _apply_detector_axis_ranges(self) -> None:
        """Validate and retain the current mode's manual X/Y ranges."""

        from tkinter import messagebox

        try:
            self._detector_axis_limits()
            self._store_detector_axis_ranges()
            self._refresh_detector_plot()
        except Exception as exc:
            messagebox.showerror("Detector 축 범위 오류", str(exc))

    def _reset_detector_axis_ranges(self) -> None:
        """Clear only the current detector mode's manual range values."""

        for variable in (
            self.detector_x_min_var,
            self.detector_x_max_var,
            self.detector_y_min_var,
            self.detector_y_max_var,
        ):
            variable.set("")
        self._store_detector_axis_ranges()
        self._refresh_detector_plot()

    def _refresh_detector_plot(self) -> None:
        """Redraw the selected cached Detector result without changing tabs."""

        from tkinter import messagebox

        result = self.last_detector_result
        if result is None:
            return
        try:
            self._render_detector_result(result)
        except Exception as exc:
            messagebox.showerror("Active Detector 그래프 오류", str(exc))

    def _render_detector_result(self, result: DetectorResult) -> None:
        """Render four traces, gate/level guides, crossings, and measurement text."""

        from matplotlib.figure import Figure

        configure_plot_fonts()

        self._dispose_cursor_group(self.detector_plot_cursors)
        self.detector_plot_cursors.clear()
        self._clear_frame(self.detector_plot_frame)
        controls = self.ttk.Frame(self.detector_plot_frame, padding=(8, 5))
        controls.pack(fill="x")
        self.ttk.Label(controls, text="표시 모드").pack(side="left")
        mode_combo = self.ttk.Combobox(
            controls,
            textvariable=self.detector_mode_var,
            values=("절대전압", "AC 비교"),
            state="readonly",
            width=10,
        )
        mode_combo.pack(side="left", padx=(4, 12))
        mode_combo.bind("<<ComboboxSelected>>", self._on_detector_mode_changed)
        for label, variable in (
            ("X min [ms]", self.detector_x_min_var),
            ("X max [ms]", self.detector_x_max_var),
            ("Y min [mV]", self.detector_y_min_var),
            ("Y max [mV]", self.detector_y_max_var),
        ):
            self.ttk.Label(controls, text=label).pack(side="left", padx=(4, 2))
            self.ttk.Entry(controls, textvariable=variable, width=8).pack(
                side="left"
            )
        self.ttk.Button(
            controls,
            text="범위 적용",
            command=self._apply_detector_axis_ranges,
        ).pack(side="left", padx=(6, 0))
        self.ttk.Button(
            controls,
            text="자동 범위",
            command=self._reset_detector_axis_ranges,
        ).pack(side="left", padx=(5, 0))

        plot_host = self.ttk.Frame(self.detector_plot_frame)
        plot_host.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        figure = Figure(figsize=(13.0, 7.4), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        mode = self.detector_mode_var.get()
        if mode == "AC 비교":
            display_rows, _baselines = detector_ac_comparison_rows(
                result.rows,
                result.settings.gate_on_s,
            )
            labels = ("v_in", "v_filt", "v_env", "v_thr relative")
        else:
            display_rows = list(result.rows)
            labels = ("v_in", "v_filt", "v_env", "v_thr")
        times_ms = [row[0] * 1000.0 for row in display_rows]
        colors = ("#1565c0", "#00897b", "#c62828", "#6a1b9a")
        data_lines: list[object] = []
        all_values_mv: list[float] = []
        for value_index, (label, color) in enumerate(
            zip(labels, colors),
            start=1,
        ):
            values_mv = [row[value_index] * 1000.0 for row in display_rows]
            all_values_mv.extend(values_mv)
            data_lines.append(
                axes.plot(times_ms, values_mv, label=label, color=color)[0]
            )
        gate_on_ms = result.settings.gate_on_s * 1000.0
        gate_off_ms = result.settings.gate_off_s * 1000.0
        axes.axvline(
            gate_on_ms,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label="Gate ON",
        )
        axes.axvline(
            gate_off_ms,
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label="Gate OFF",
        )
        x_min, x_max, y_min, y_max = self._detector_axis_limits()
        auto_x_min, auto_x_max = min(times_ms), max(times_ms)
        auto_y_min, auto_y_max = data_occupancy_limits(all_values_mv)
        axes.set_xlim(
            left=auto_x_min if x_min is None else x_min,
            right=auto_x_max if x_max is None else x_max,
        )
        axes.set_ylim(
            bottom=auto_y_min if y_min is None else y_min,
            top=auto_y_max if y_max is None else y_max,
        )
        stale = result.is_stale(self.component_revision)
        title_suffix = " - 소자 변경 전 결과, 재실행 필요" if stale else ""
        axes.set_title(
            f"Active Detector ch{result.channel.ch:02d} "
            f"({result.channel.detector_opamp}) - {mode}{title_suffix}"
        )
        axes.set_xlabel("Time [ms]")
        axes.set_ylabel("Voltage [mV]")
        axes.grid(True, which="both", alpha=0.28)
        axes.legend(fontsize=8, ncol=3)
        self.detector_canvas = self._embed_detector_figure(
            plot_host,
            figure,
            axes,
            data_lines,
        )

    def _embed_detector_figure(
        self,
        parent: object,
        figure: object,
        axes: object,
        lines: list[object],
    ) -> object:
        """Embed the large Detector graph with the reusable line cursor."""

        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
        except ImportError as exc:
            raise RuntimeError("그래프를 표시하려면 matplotlib가 필요합니다.") from exc
        canvas = FigureCanvasTkAgg(figure, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        cursor = InteractivePlotCursor(
            canvas,
            axes,
            lines,
            x_name="Time",
            x_unit="ms",
            y_name="Voltage",
            y_unit="mV",
        )
        cursor_controls = self.ttk.Frame(parent)
        cursor_controls.pack(side="bottom", fill="x", pady=(2, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 추가",
            command=cursor.arm,
        ).pack(side="left")
        self.ttk.Button(
            cursor_controls,
            text="마지막 커서 삭제",
            command=cursor.remove_last,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 전체 삭제",
            command=cursor.clear,
        ).pack(side="left", padx=(5, 0))
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.detector_plot_cursors.append(cursor)
        return canvas
