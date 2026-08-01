"""AC controls, magnitude/phase plots, metrics, and selection overlay."""

from __future__ import annotations

import math
import re
from typing import Callable, Sequence

from ..models import AcMetrics, ChannelResult
from ..results import format_metric_copy_line
from ..values import (
    data_occupancy_limits,
    db_to_linear,
    linear_magnitude_auto_limits,
)
from .base import AppController, configure_plot_fonts
from .cursors import InteractivePlotCursor, MetricSelectionOverlay


class AcTabController(AppController):
    """AC controls, magnitude/phase plots, metrics, and selection overlay."""

    def _build_ac_tab(self) -> None:
        """Build AC controls plus magnitude, phase, and log result tabs."""

        outer = self.ttk.Frame(self.ac_page, padding=10)
        outer.pack(fill="both", expand=True)
        settings_row = self.ttk.Frame(outer)
        settings_row.pack(fill="x")
        settings_row.columnconfigure(0, weight=5)
        settings_row.columnconfigure(1, weight=3)
        settings_row.columnconfigure(2, weight=2)
        files = self._build_file_settings(settings_row)
        files.grid(row=0, column=0, sticky="nsew")

        sweep = self.ttk.LabelFrame(settings_row, text="AC 해석 설정", padding=8)
        sweep.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        fields = (
            ("AC points/dec", self.ac_points_var),
            ("AC start [Hz]", self.ac_start_var),
            ("AC stop [Hz]", self.ac_stop_var),
            ("AC 출력 노드", self.output_node_var),
        )
        for row, (label, variable) in enumerate(fields):
            self.ttk.Label(sweep, text=label).grid(
                row=row, column=0, sticky="e", padx=(4, 3), pady=3
            )
            self.ttk.Entry(sweep, textvariable=variable, width=16).grid(
                row=row, column=1, columnspan=2, sticky="ew", padx=(0, 7), pady=3
            )
        sweep.columnconfigure(1, weight=1)
        self.ttk.Checkbutton(
            sweep,
            text="TI PSpice + KiCad 호환 모드 (ngbehavior=pski)",
            variable=self.pspice_compat_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(5, 0))
        channels_frame = self._build_channel_selector(
            settings_row,
            "ac_channel_list",
            self._select_all_ac,
            self._clear_ac_selection,
            title="AC 채널",
        )
        channels_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=8)
        self.ac_run_button = self.ttk.Button(
            action,
            text="AC 시뮬 실행",
            command=lambda: self._start_run(("ac",), self.ac_channel_list),
        )
        self.ac_run_button.pack(side="left")
        self.ac_stop_button = self.ttk.Button(
            action, text="중지", command=self._stop_run, state="disabled"
        )
        self.ac_stop_button.pack(side="left", padx=(6, 0))
        self.ttk.Button(
            action, text="결과 폴더 열기", command=self._open_results
        ).pack(side="left", padx=(6, 0))
        self.ac_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=180
        )
        self.ac_progress.pack(side="left", padx=12)
        self.ttk.Label(action, textvariable=self.status_var).pack(
            side="left", fill="x"
        )

        self.ac_result_notebook = self.ttk.Notebook(outer)
        self.ac_result_notebook.pack(fill="both", expand=True)
        self.mag_frame = self.ttk.Frame(self.ac_result_notebook)
        self.phase_frame = self.ttk.Frame(self.ac_result_notebook)
        self.log_frame = self.ttk.Frame(self.ac_result_notebook)
        self.ac_result_notebook.add(self.mag_frame, text="AC Magnitude")
        self.ac_result_notebook.add(self.phase_frame, text="AC Phase")
        self.ac_result_notebook.add(self.log_frame, text="실행 로그")
        self.log_text = self.tk.Text(self.log_frame, wrap="none", height=15)
        log_y = self.ttk.Scrollbar(
            self.log_frame, orient="vertical", command=self.log_text.yview
        )
        log_x = self.ttk.Scrollbar(
            self.log_frame, orient="horizontal", command=self.log_text.xview
        )
        self.log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")
        self.log_frame.rowconfigure(0, weight=1)
        self.log_frame.columnconfigure(0, weight=1)
        # Compatibility aliases retained for extension code written against v8.
        self.run_button = self.ac_run_button
        self.stop_button = self.ac_stop_button
        self.progress = self.ac_progress
        self.notebook = self.ac_result_notebook

    def _select_all_ac(self) -> None:
        """Select every loaded channel in the AC run list."""

        self.ac_channel_list.selection_set(0, "end")

    def _clear_ac_selection(self) -> None:
        """Clear the AC run-list channel selection."""

        self.ac_channel_list.selection_clear(0, "end")

    def _refresh_magnitude_tab(self) -> None:
        """Validate controls and redraw only the AC magnitude tab."""

        from tkinter import messagebox

        try:
            self._magnitude_axis_limits()
        except ValueError as exc:
            messagebox.showerror("AC Magnitude 축 범위 오류", str(exc))
            return
        self._store_magnitude_y_limits()
        self._dispose_cursor_group(self.ac_plot_cursors["magnitude"])
        self.ac_plot_cursors["magnitude"].clear()
        self._clear_frame(self.mag_frame)
        ac_results = list(self.last_ac_results)
        if ac_results:
            self._render_magnitude(ac_results)
        else:
            self.ttk.Label(
                self.mag_frame, text="이번 실행에는 AC 결과가 없습니다."
            ).pack(expand=True)

    def _refresh_phase_tab(self) -> None:
        """Redraw only the AC phase tab after an axis selection changes."""

        self._dispose_cursor_group(self.ac_plot_cursors["phase"])
        self.ac_plot_cursors["phase"].clear()
        self._clear_frame(self.phase_frame)
        ac_results = list(self.last_ac_results)
        if ac_results:
            self._render_phase(ac_results)
        else:
            self.ttk.Label(
                self.phase_frame, text="이번 실행에는 AC 결과가 없습니다."
            ).pack(expand=True)

    def _plot_controls(
        self,
        parent: object,
        x_variable: object,
        y_variable: object,
        y_values: Sequence[str],
        refresh_command: Callable[[], None],
        range_variables: tuple[object, object, object, object] | None = None,
        y_label: str = "Y축",
        y_change_command: Callable[[], None] | None = None,
    ) -> None:
        """Build axis-scale/range controls shared by AC plot tabs."""

        controls = self.ttk.Frame(parent, padding=(8, 6))
        controls.pack(fill="x")
        top = self.ttk.Frame(controls)
        top.pack(fill="x")
        self.ttk.Label(top, text="X축").pack(side="left")
        x_combo = self.ttk.Combobox(
            top,
            textvariable=x_variable,
            values=("Linear", "Decade"),
            state="readonly",
            width=10,
        )
        x_combo.pack(side="left", padx=(4, 14))
        self.ttk.Label(top, text=y_label).pack(side="left")
        y_combo = self.ttk.Combobox(
            top,
            textvariable=y_variable,
            values=tuple(y_values),
            state="readonly" if len(y_values) > 1 else "disabled",
            width=17,
        )
        y_combo.pack(side="left", padx=(4, 14))
        if range_variables is not None:
            x_min, x_max, y_min, y_max = range_variables
            self.ttk.Separator(top, orient="vertical").pack(
                side="left", fill="y", padx=(0, 12)
            )
            self.ttk.Label(top, text="X 범위").pack(side="left")
            self.ttk.Entry(top, textvariable=x_min, width=9).pack(
                side="left", padx=(4, 2)
            )
            self.ttk.Label(top, text="~").pack(side="left")
            self.ttk.Entry(top, textvariable=x_max, width=9).pack(
                side="left", padx=(2, 12)
            )
            self.ttk.Label(top, text="Y 범위").pack(side="left")
            self.ttk.Entry(top, textvariable=y_min, width=9).pack(
                side="left", padx=(4, 2)
            )
            self.ttk.Label(top, text="~").pack(side="left")
            self.ttk.Entry(top, textvariable=y_max, width=9).pack(
                side="left", padx=(2, 8)
            )
            self.ttk.Button(
                top, text="범위 적용", command=refresh_command
            ).pack(side="left")
            self.ttk.Button(
                top, text="자동 범위", command=self._reset_magnitude_axis_limits
            ).pack(side="left", padx=(5, 0))
        help_prefix = (
            "범위 입력은 현재 축 단위 기준, 빈칸은 자동 · "
            if range_variables is not None
            else ""
        )
        self.ttk.Label(
            controls,
            text=help_prefix
            + (
                "커서 추가 → 선을 따라 이동 → 클릭해 고정 · 반복하면 다중 커서 · "
                "아래 도구: 확대/이동/저장"
            ),
        ).pack(anchor="w", pady=(5, 0))
        x_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: refresh_command(),
        )
        y_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: (
                y_change_command() if y_change_command else refresh_command()
            ),
        )

    @staticmethod
    def _parse_axis_limits(
        minimum_text: str,
        maximum_text: str,
        axis_name: str,
        logarithmic: bool,
    ) -> tuple[float | None, float | None]:
        """Parse optional axis bounds and enforce ordering/log constraints."""

        def parse_one(text: str, field_name: str) -> float | None:
            """Parse one optional finite bound for the enclosing axis."""

            stripped = text.strip()
            if not stripped:
                return None
            try:
                value = float(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name}에는 숫자 또는 빈칸을 입력하세요."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{field_name}에는 유한한 숫자를 입력하세요.")
            if logarithmic and value <= 0:
                raise ValueError(
                    f"{axis_name}이 Decade일 때 {field_name}은 0보다 커야 합니다."
                )
            return value

        minimum = parse_one(minimum_text, f"{axis_name} 최소")
        maximum = parse_one(maximum_text, f"{axis_name} 최대")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(
                f"{axis_name} 최소값은 최대값보다 작아야 합니다."
            )
        return minimum, maximum

    def _magnitude_axis_limits(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Read and validate the four manual magnitude-axis bounds."""

        x_min, x_max = self._parse_axis_limits(
            self.mag_x_min_var.get(),
            self.mag_x_max_var.get(),
            "X축",
            self.mag_x_scale_var.get() == "Decade",
        )
        y_min, y_max = self._parse_axis_limits(
            self.mag_y_min_var.get(),
            self.mag_y_max_var.get(),
            "Y축",
            False,
        )
        return x_min, x_max, y_min, y_max

    def _store_magnitude_y_limits(self, unit: str | None = None) -> None:
        """Save the visible Y text under its dB or Linear display unit."""

        active = unit or self.mag_unit_var.get()
        if active not in {"dB", "Linear"}:
            return
        self.ac_manual_y_limits[active] = (
            self.mag_y_min_var.get(),
            self.mag_y_max_var.get(),
        )

    def _restore_magnitude_y_limits(self, unit: str) -> None:
        """Restore unit-specific manual Y bounds without converting them."""

        minimum, maximum = self.ac_manual_y_limits.get(unit, ("", ""))
        self.mag_y_min_var.set(minimum)
        self.mag_y_max_var.set(maximum)

    def _on_magnitude_unit_changed(self) -> None:
        """Switch dB/Linear data while preserving two independent Y ranges."""

        previous = getattr(self, "mag_previous_unit", "dB")
        self._store_magnitude_y_limits(previous)
        current = self.mag_unit_var.get()
        if current not in {"dB", "Linear"}:
            self.mag_unit_var.set(previous)
            return
        self.ac_magnitude_unit = current
        self._restore_magnitude_y_limits(current)
        self.mag_previous_unit = current
        self._refresh_magnitude_tab()

    def _reset_magnitude_axis_limits(self) -> None:
        """Clear all manual magnitude limits and redraw with autoscaling."""

        for variable in (
            self.mag_x_min_var,
            self.mag_x_max_var,
            self.mag_y_min_var,
            self.mag_y_max_var,
        ):
            variable.set("")
        self._store_magnitude_y_limits()
        self._refresh_magnitude_tab()

    def _embed_figure(
        self,
        parent: object,
        figure: object,
        axes: object,
        data_lines: Sequence[object],
        cursor_group: str,
        *,
        x_name: str = "x",
        x_unit: str = "",
        y_name: str = "y",
        y_unit: str = "",
    ) -> object:
        """Embed a figure and register its cursor under one AC plot group."""

        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
        except ImportError as exc:
            raise RuntimeError(
                "그래프를 표시하려면 matplotlib가 필요합니다."
            ) from exc
        host = self.ttk.Frame(parent)
        host.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        canvas = FigureCanvasTkAgg(figure, master=host)
        toolbar = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        cursor = InteractivePlotCursor(
            canvas,
            axes,
            data_lines,
            x_name=x_name,
            x_unit=x_unit,
            y_name=y_name,
            y_unit=y_unit,
        )
        cursor_controls = self.ttk.Frame(host)
        cursor_controls.pack(side="bottom", fill="x", pady=(2, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 추가",
            command=cursor.arm,
        ).pack(side="left")
        self.ttk.Button(
            cursor_controls,
            text="커서 전체 삭제",
            command=cursor.clear,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Label(
            cursor_controls,
            text="추가 → 선을 따라 이동 → 세로 보조선 위치 클릭 · 반복 추가 가능",
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.ac_plot_cursors.setdefault(cursor_group, []).append(cursor)
        return canvas

    def _render_magnitude(self, results: Sequence[ChannelResult]) -> None:
        """Render the magnitude graph left and the 16-channel metrics list right."""

        from matplotlib.figure import Figure

        configure_plot_fonts()

        self._plot_controls(
            self.mag_frame,
            self.mag_x_scale_var,
            self.mag_unit_var,
            ("dB", "Linear"),
            self._refresh_magnitude_tab,
            (
                self.mag_x_min_var,
                self.mag_x_max_var,
                self.mag_y_min_var,
                self.mag_y_max_var,
            ),
            y_label="Magnitude 단위",
            y_change_command=self._on_magnitude_unit_changed,
        )
        paned = self.ttk.Panedwindow(self.mag_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        plot_host = self.ttk.Frame(paned)
        metrics_host = self.ttk.Frame(paned, width=390)
        paned.add(plot_host, weight=4)
        paned.add(metrics_host, weight=1)
        figure = Figure(figsize=(12.5, 7.2), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        use_decade_x = self.mag_x_scale_var.get() == "Decade"
        magnitude_unit = self.mag_unit_var.get()
        x_min, x_max, y_min, y_max = self._magnitude_axis_limits()
        data_lines: list[object] = []
        all_y_values: list[float] = []
        metric_entries: dict[int, tuple[object, AcMetrics, float]] = {}
        for result in results:
            label = f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)"
            frequencies = [row[0] for row in result.ac_rows]
            db_values = [row[1] for row in result.ac_rows]
            y_values = (
                [db_to_linear(value) for value in db_values]
                if magnitude_unit == "Linear"
                else db_values
            )
            all_y_values.extend(y_values)
            line = axes.plot(frequencies, y_values, label=label)[0]
            data_lines.append(line)
            metric = result.ac_metrics
            if metric is not None:
                marker_y = (
                    db_to_linear(metric.peak_db)
                    if magnitude_unit == "Linear"
                    else metric.peak_db
                )
                axes.plot(
                    metric.center_hz,
                    marker_y,
                    marker="o",
                    markersize=5,
                    color=line.get_color(),
                    label="_nolegend_",
                )
                metric_entries[result.channel.ch] = (line, metric, marker_y)
        if use_decade_x:
            axes.set_xscale("log")
        if magnitude_unit == "Linear":
            axes.set_ylabel("Magnitude [Linear V/V]")
        else:
            axes.set_ylabel("Magnitude [dB]")
        axes.set_xlim(left=x_min, right=x_max)
        if magnitude_unit == "Linear":
            auto_y_min, auto_y_max = linear_magnitude_auto_limits(all_y_values)
        else:
            auto_y_min, auto_y_max = data_occupancy_limits(all_y_values)
        resolved_y_min = auto_y_min if y_min is None else y_min
        resolved_y_max = auto_y_max if y_max is None else y_max
        if resolved_y_min >= resolved_y_max:
            raise ValueError(
                "수동 Y축 경계와 자동 Y축 경계를 함께 적용한 결과 "
                "최소값이 최대값 이상입니다."
            )
        axes.set_ylim(bottom=resolved_y_min, top=resolved_y_max)
        axes.set_title("AC Magnitude")
        axes.set_xlabel("Frequency [Hz]")
        axes.grid(True, which="both", alpha=0.32)
        axes.legend(fontsize=8, ncol=2)
        canvas = self._embed_figure(
            plot_host,
            figure,
            axes,
            data_lines,
            "magnitude",
            x_name="Frequency",
            x_unit="Hz",
            y_name="Gain",
            y_unit=magnitude_unit,
        )
        self.metric_selection_overlay = MetricSelectionOverlay(
            canvas,
            axes,
            metric_entries,
            gain_unit=magnitude_unit,
        )
        self._build_metrics_table(
            metrics_host, results, self.metric_selection_overlay
        )

    def _render_phase(self, results: Sequence[ChannelResult]) -> None:
        """Render all selected AC phase curves with interactive inspection."""

        from matplotlib.figure import Figure

        configure_plot_fonts()

        self._plot_controls(
            self.phase_frame,
            self.phase_x_scale_var,
            self.phase_y_scale_var,
            ("Linear (deg)",),
            self._refresh_phase_tab,
        )
        figure = Figure(figsize=(12.5, 7.8), dpi=100, constrained_layout=True)
        axes = figure.add_subplot(111)
        data_lines: list[object] = []
        for result in results:
            label = f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)"
            line = axes.plot(
                [row[0] for row in result.ac_rows],
                [row[2] for row in result.ac_rows],
                label=label,
            )[0]
            data_lines.append(line)
        if self.phase_x_scale_var.get() == "Decade":
            axes.set_xscale("log")
        axes.set_title("AC Phase")
        axes.set_xlabel("Frequency [Hz]")
        axes.set_ylabel("Phase [deg]")
        axes.grid(True, which="both", alpha=0.32)
        axes.legend(fontsize=8, ncol=2)
        self._embed_figure(
            self.phase_frame,
            figure,
            axes,
            data_lines,
            "phase",
            x_name="Frequency",
            x_unit="Hz",
            y_name="Phase",
            y_unit="deg",
        )

    def _build_metrics_table(
        self,
        parent: object,
        results: Sequence[ChannelResult],
        overlay: MetricSelectionOverlay,
    ) -> None:
        """Build the copyable 16-row list that controls persistent peak cursors."""

        frame = self.ttk.LabelFrame(
            parent, text="16채널 f0 / Gain / Q", padding=6
        )
        frame.pack(fill="both", expand=True)
        self.ttk.Label(
            frame,
            text=(
                "행 클릭: 선택 추가/해제\n"
                "여러 행을 선택하면 그래프에 여러 커서가 표시됩니다."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 5))
        columns = ("channel", "center", "gain", "q")
        table = self.ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=16,
        )
        magnitude_unit = self.mag_unit_var.get()
        headings = {
            "channel": "채널",
            "center": "중심주파수 [Hz]",
            "gain": f"Gain [{magnitude_unit}]",
            "q": "Q",
        }
        widths = {
            "channel": 58,
            "center": 125,
            "gain": 92,
            "q": 72,
        }
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(
                column,
                width=widths[column],
                anchor="center",
                stretch=column == "center",
            )
        result_by_channel = {
            result.channel.ch: result for result in results
        }
        channels = self.channels or [
            result.channel for result in sorted(
                results, key=lambda item: item.channel.ch
            )
        ]
        self.metric_copy_rows = {}
        for channel in sorted(channels, key=lambda item: item.ch):
            result = result_by_channel.get(channel.ch)
            metric = result.ac_metrics if result else None
            item_id = f"ch{channel.ch:02d}"
            table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    item_id,
                    self._format_metric(metric.center_hz if metric else None),
                    self._format_metric(
                        (
                            db_to_linear(metric.peak_db)
                            if metric and magnitude_unit == "Linear"
                            else metric.peak_db if metric else None
                        ),
                        digits=4,
                    ),
                    self._format_metric(metric.q if metric else None, digits=4),
                ),
            )
            self.metric_copy_rows[item_id] = format_metric_copy_line(
                channel,
                metric,
                magnitude_unit,
            )
        scroll = self.ttk.Scrollbar(
            frame, orient="vertical", command=table.yview
        )
        table.configure(yscrollcommand=scroll.set)
        table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        buttons = self.ttk.Frame(parent, padding=(0, 6, 0, 0))
        buttons.pack(fill="x")
        self.ttk.Button(
            buttons,
            text="선택 행 복사",
            command=lambda: self._copy_metric_rows(False),
        ).pack(side="left")
        self.ttk.Button(
            buttons,
            text="전체 16채널 복사",
            command=lambda: self._copy_metric_rows(True),
        ).pack(side="left", padx=(5, 0))
        table.bind(
            "<Button-1>",
            lambda event: self._toggle_metric_table_row(
                event, table, overlay
            ),
        )
        table.bind(
            "<<TreeviewSelect>>",
            lambda _event: self._sync_metric_selection(table, overlay),
        )
        table.bind(
            "<Control-c>",
            lambda _event: self._copy_metric_rows(False),
        )
        table.bind(
            "<Control-C>",
            lambda _event: self._copy_metric_rows(False),
        )
        self.metric_table = table
        for channel in sorted(self.selected_metric_channels):
            item_id = f"ch{channel:02d}"
            if table.exists(item_id):
                table.selection_add(item_id)
        self._sync_metric_selection(table, overlay)

    def _toggle_metric_table_row(
        self,
        event: object,
        table: object,
        overlay: MetricSelectionOverlay,
    ) -> str | None:
        """Toggle one metric row on a plain click so Ctrl is not required."""

        if table.identify_region(event.x, event.y) not in {"cell", "tree"}:
            return None
        item_id = table.identify_row(event.y)
        if not item_id:
            return None
        if item_id in table.selection():
            table.selection_remove(item_id)
        else:
            table.selection_add(item_id)
        table.focus(item_id)
        self._sync_metric_selection(table, overlay)
        return "break"

    def _sync_metric_selection(
        self,
        table: object,
        overlay: MetricSelectionOverlay,
    ) -> None:
        """Mirror selected metric rows into the persistent graph overlay."""

        channel_numbers = []
        for item_id in table.selection():
            match = re.fullmatch(r"ch(\d+)", item_id)
            if match:
                channel_numbers.append(int(match.group(1)))
        self.selected_metric_channels = set(channel_numbers)
        overlay.show_channels(sorted(channel_numbers))

    def _copy_metric_rows(self, copy_all: bool) -> str:
        """Copy selected or all metric lines as tab-separated plain text."""

        table = self.metric_table
        if table is None:
            return "break"
        item_ids = (
            tuple(table.get_children())
            if copy_all
            else tuple(table.selection())
        )
        if not item_ids:
            item_ids = tuple(table.get_children())
        text = "\n".join(
            self.metric_copy_rows[item_id]
            for item_id in item_ids
            if item_id in self.metric_copy_rows
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_var.set(f"AC 메트릭 {len(item_ids)}행 복사됨")
        return "break"

    @staticmethod
    def _format_metric(value: float | None, digits: int = 6) -> str:
        """Format an optional finite metric or return an em dash."""

        if value is None or not math.isfinite(value):
            return "—"
        return f"{value:.{digits}g}"
