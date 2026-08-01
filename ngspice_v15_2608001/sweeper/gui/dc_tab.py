"""DC operating-point schematic, component editor, and view state."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from ..components import (
    apply_component_updates,
    divider_resistors_for_vref,
    divider_vref_v,
    vref_from_margin_v,
)
from ..constants import (
    COMPONENT_EDIT_FIELDS,
    DEFAULT_DETECTOR_OPAMP,
    DETECTOR_OPAMP_NAMES,
    DEFAULT_SCHEMATIC,
    DEFAULT_TABLE,
    RESISTANCE_EDIT_KEYS,
    SCHEMATIC_COMPONENT_POSITIONS,
    SCHEMATIC_DETECTOR_OPAMP_POSITION,
    SCHEMATIC_VALUE_FONT_SIZE,
)
from ..models import Channel, ChannelResult
from ..results import _canonical_vector_name
from ..values import (
    format_margin_mv,
    format_millivolts,
    format_resistance_kohm,
    format_voltage_mv,
    parallel_resistance_kohm,
    round_resistance_kohm,
    voltage_margin_v,
)
from .base import AppController


class DcTabController(AppController):
    """DC operating-point schematic, component editor, and view state."""

    def _build_dc_tab(self) -> None:
        """Build DC operating-point controls and its full-width schematic view."""

        outer = self.ttk.Frame(self.dc_page, padding=10)
        outer.pack(fill="both", expand=True)
        settings_row = self.ttk.Frame(outer)
        settings_row.pack(fill="x")
        settings_row.columnconfigure(0, weight=5)
        settings_row.columnconfigure(1, weight=2)
        settings_row.columnconfigure(2, weight=2)
        files = self._build_file_settings(settings_row)
        files.grid(row=0, column=0, sticky="nsew")
        dc_settings = self.ttk.LabelFrame(
            settings_row, text="DC 동작점 설정", padding=8
        )
        dc_settings.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.ttk.Checkbutton(
            dc_settings,
            text="TI PSpice + KiCad 호환 모드\n(ngbehavior=pski)",
            variable=self.pspice_compat_var,
        ).pack(anchor="w")
        self.ttk.Label(
            dc_settings,
            text=(
                ".save all + .op를 실행합니다.\n"
                "소자 편집의 변경값 적용도 현재 채널의\n"
                "DC 동작점을 자동 재실행합니다."
            ),
            justify="left",
            foreground="#164a7b",
        ).pack(anchor="w", pady=(8, 0))
        channels_frame = self._build_channel_selector(
            settings_row,
            "channel_list",
            self._select_all,
            self._clear_selection,
            title="DC 채널",
            on_select=self._on_dc_channel_selection_changed,
        )
        channels_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=8)
        self.op_run_button = self.ttk.Button(
            action,
            text="DC 동작점 실행",
            command=lambda: self._start_run(("op",), self.channel_list),
        )
        self.op_run_button.pack(side="left")
        self.op_stop_button = self.ttk.Button(
            action, text="중지", command=self._stop_run, state="disabled"
        )
        self.op_stop_button.pack(side="left", padx=(6, 0))
        self.ttk.Button(
            action, text="결과 폴더 열기", command=self._open_results
        ).pack(side="left", padx=(6, 0))
        self.op_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=180
        )
        self.op_progress.pack(side="left", padx=12)
        self.ttk.Label(action, textvariable=self.status_var).pack(
            side="left", fill="x"
        )
        self.op_frame = self.ttk.Frame(outer)
        self.op_frame.pack(fill="both", expand=True)

    def _on_dc_channel_selection_changed(self) -> None:
        """Follow the clicked DC channel in the schematic and component editor.

        The DC editor belongs to the displayed operating-point result, so the
        list selection is forwarded to that result's channel combo.  Channels
        without a result yet are ignored because there is nothing to show.
        """

        selected = self._channels_from_listbox("channel_list")
        if not selected:
            return
        first = selected[0].ch
        if self.component_editor_channel == first:
            return
        label = next(
            (
                f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)"
                for result in self.last_op_results
                if result.channel.ch == first
            ),
            None,
        )
        if label is None or self.selected_op_channel_var.get() == label:
            return
        self.selected_op_channel_var.set(label)
        self._on_op_channel_changed(None)

    def _refresh_operating_point_tab(self) -> None:
        """Redraw only the OP schematic tab, preserving the outer tab selection."""

        self._capture_operating_point_view_state()
        self._clear_frame(self.op_frame)
        op_results = list(self.last_op_results)
        if op_results:
            self._render_operating_point(op_results)
        else:
            self.ttk.Label(
                self.op_frame, text="이번 실행에는 DC 동작점 결과가 없습니다."
            ).pack(expand=True)

    def _capture_operating_point_view_state(self) -> None:
        """Save the OP detail tab, pane ratio, and schematic scroll position."""

        notebook = self.op_detail_notebook
        if notebook is not None:
            try:
                selected = notebook.select()
                for name, tab in self.op_detail_tabs.items():
                    if str(tab) == selected:
                        self.op_detail_selected_tab = name
                        break
            except Exception:
                pass

        paned = self.op_paned
        if paned is not None:
            try:
                paned.update_idletasks()
                width = int(paned.winfo_width())
                if width > 1:
                    sash = int(paned.sashpos(0))
                    self.op_pane_fraction = max(
                        0.05,
                        min(0.95, sash / width),
                    )
            except Exception:
                pass

        canvas = self.schematic_canvas
        if canvas is not None:
            try:
                self.op_canvas_view = (
                    float(canvas.xview()[0]),
                    float(canvas.yview()[0]),
                )
            except Exception:
                pass

    def _restore_operating_point_view_state(self, paned: object) -> None:
        """Restore saved OP geometry after Tk has laid out replacement widgets."""

        if paned is not self.op_paned:
            return
        notebook = self.op_detail_notebook
        selected_detail = self.op_detail_tabs.get(
            self.op_detail_selected_tab
        )
        if notebook is not None and selected_detail is not None:
            try:
                notebook.select(selected_detail)
            except Exception:
                pass
        try:
            paned.update_idletasks()
            width = int(paned.winfo_width())
            if self.op_pane_fraction is not None and width > 1:
                paned.sashpos(0, round(width * self.op_pane_fraction))
        except Exception:
            pass
        canvas = self.schematic_canvas
        if canvas is not None and self.op_canvas_view is not None:
            try:
                canvas.update_idletasks()
                canvas.xview_moveto(self.op_canvas_view[0])
                canvas.yview_moveto(self.op_canvas_view[1])
            except Exception:
                pass

    def _render_operating_point(
        self, results: Sequence[ChannelResult]
    ) -> None:
        """Render one selectable channel schematic plus OP/component detail panes."""

        labels = {
            f"ch{result.channel.ch:02d} ({result.channel.f_c_hz:g} Hz)": result
            for result in results
        }
        if self.selected_op_channel_var.get() not in labels:
            self.selected_op_channel_var.set(next(iter(labels)))
        result = labels[self.selected_op_channel_var.get()]

        controls = self.ttk.Frame(self.op_frame, padding=(8, 6))
        controls.pack(fill="x")
        self.ttk.Label(controls, text="채널").pack(side="left")
        channel_combo = self.ttk.Combobox(
            controls,
            textvariable=self.selected_op_channel_var,
            values=tuple(labels),
            state="readonly",
            width=24,
        )
        channel_combo.pack(side="left", padx=(4, 16))
        channel_combo.bind(
            "<<ComboboxSelected>>",
            self._on_op_channel_changed,
        )
        self.ttk.Label(controls, text="회로도 확대").pack(side="left")
        self.ttk.Button(
            controls,
            text="−",
            width=3,
            command=lambda: self._change_schematic_zoom(1.0 / 1.2),
        ).pack(side="left", padx=(5, 2))
        zoom = self.ttk.Scale(
            controls,
            from_=0.5,
            to=2.5,
            variable=self.schematic_zoom_var,
            length=150,
        )
        zoom.pack(side="left", padx=2)
        zoom.bind(
            "<ButtonRelease-1>",
            lambda _event: self._redraw_current_schematic(),
        )
        self.ttk.Button(
            controls,
            text="+",
            width=3,
            command=lambda: self._change_schematic_zoom(1.2),
        ).pack(side="left", padx=(2, 4))
        self.ttk.Button(
            controls,
            text="100%",
            command=lambda: self._set_schematic_zoom(1.0),
        ).pack(side="left", padx=(0, 5))
        self.ttk.Label(
            controls,
            textvariable=self.schematic_zoom_var,
            width=5,
        ).pack(side="left")
        self.ttk.Label(
            controls,
            text=(
                "파랑: OP 노드 전압 [mV] · 초록: 적용 소자값 · "
                "휠: 확대/축소 · 좌클릭 드래그: 이동"
            ),
        ).pack(side="left", padx=(16, 0))

        paned = self.ttk.Panedwindow(self.op_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        drawing_host = self.ttk.Frame(paned)
        details_host = self.ttk.Frame(paned, width=330)
        paned.add(drawing_host, weight=4)
        paned.add(details_host, weight=1)
        self.op_paned = paned

        canvas = self.tk.Canvas(drawing_host, background="#ececec")
        x_scroll = self.ttk.Scrollbar(
            drawing_host, orient="horizontal", command=canvas.xview
        )
        y_scroll = self.ttk.Scrollbar(
            drawing_host, orient="vertical", command=canvas.yview
        )
        canvas.configure(
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        drawing_host.rowconfigure(0, weight=1)
        drawing_host.columnconfigure(0, weight=1)
        self.schematic_canvas = canvas
        self.schematic_result = result
        canvas.bind("<ButtonPress-1>", self._start_schematic_pan)
        canvas.bind("<B1-Motion>", self._drag_schematic)
        canvas.bind("<ButtonRelease-1>", self._stop_schematic_pan)
        canvas.bind("<MouseWheel>", self._zoom_schematic_with_wheel)
        canvas.bind("<Button-4>", self._zoom_schematic_with_wheel)
        canvas.bind("<Button-5>", self._zoom_schematic_with_wheel)
        self._draw_operating_point_schematic(canvas, result)
        self._build_op_detail_tables(details_host, result)
        self.root.after_idle(
            lambda current=paned: self._restore_operating_point_view_state(
                current
            )
        )

    @staticmethod
    def _start_schematic_pan(event: object) -> None:
        """Start Canvas scan-based panning at the pressed mouse position."""

        canvas = event.widget
        canvas.scan_mark(event.x, event.y)
        canvas.configure(cursor="fleur")

    @staticmethod
    def _drag_schematic(event: object) -> None:
        """Pan the enlarged schematic while the left mouse button is dragged."""

        event.widget.scan_dragto(event.x, event.y, gain=1)

    @staticmethod
    def _stop_schematic_pan(event: object) -> None:
        """Restore the normal pointer after schematic dragging ends."""

        event.widget.configure(cursor="")

    def _zoom_schematic_with_wheel(self, event: object) -> str:
        """Zoom around the mouse location for Windows, macOS, and Linux wheels."""

        button = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        zoom_in = button == 4 or delta > 0
        factor = 1.15 if zoom_in else 1.0 / 1.15
        self._set_schematic_zoom(
            float(self.schematic_zoom_var.get()) * factor,
            focus=(event.x, event.y),
        )
        return "break"

    def _change_schematic_zoom(self, factor: float) -> None:
        """Apply a multiplicative zoom step from the +/- buttons."""

        self._set_schematic_zoom(float(self.schematic_zoom_var.get()) * factor)

    def _redraw_current_schematic(self) -> None:
        """Apply the scale slider value to the existing schematic canvas."""

        self._set_schematic_zoom(float(self.schematic_zoom_var.get()))

    def _set_schematic_zoom(
        self,
        requested_scale: float,
        focus: tuple[float, float] | None = None,
    ) -> None:
        """Redraw at a bounded scale while preserving the focus point."""

        canvas = self.schematic_canvas
        result = self.schematic_result
        if canvas is None or result is None or not canvas.winfo_exists():
            return
        canvas.update_idletasks()
        focus_x, focus_y = focus or (
            max(1, canvas.winfo_width()) / 2.0,
            max(1, canvas.winfo_height()) / 2.0,
        )
        region = tuple(float(value) for value in canvas.cget("scrollregion").split())
        if len(region) == 4 and region[2] > region[0] and region[3] > region[1]:
            relative_x = (canvas.canvasx(focus_x) - region[0]) / (
                region[2] - region[0]
            )
            relative_y = (canvas.canvasy(focus_y) - region[1]) / (
                region[3] - region[1]
            )
        else:
            relative_x = relative_y = 0.5
        scale = max(0.5, min(2.5, requested_scale))
        self.schematic_zoom_var.set(round(scale, 3))
        self._draw_operating_point_schematic(canvas, result)
        canvas.update_idletasks()
        new_region = tuple(
            float(value) for value in canvas.cget("scrollregion").split()
        )
        if len(new_region) == 4:
            width = max(1.0, new_region[2] - new_region[0])
            height = max(1.0, new_region[3] - new_region[1])
            target_x = relative_x * width - focus_x
            target_y = relative_y * height - focus_y
            canvas.xview_moveto(max(0.0, min(1.0, target_x / width)))
            canvas.yview_moveto(max(0.0, min(1.0, target_y / height)))

    def _draw_operating_point_schematic(
        self, canvas: object, result: ChannelResult
    ) -> None:
        """Draw the cropped PDF image with simulated mV and component overlays."""

        try:
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "회로도를 표시하려면 Pillow가 필요합니다: py -m pip install Pillow"
            ) from exc
        if not DEFAULT_SCHEMATIC.is_file():
            raise FileNotFoundError(f"회로도 이미지를 찾을 수 없습니다: {DEFAULT_SCHEMATIC}")
        scale = max(0.5, min(2.5, float(self.schematic_zoom_var.get())))
        with Image.open(DEFAULT_SCHEMATIC) as source:
            width = max(1, round(source.width * scale))
            height = max(1, round(source.height * scale))
            image = source.convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
        photo = ImageTk.PhotoImage(image)
        self.schematic_photo = photo
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")

        node_positions = (
            ("/vin", "v_in", 130, 135),
            ("Net-_V3-Pad1_", "V3+", 88, 214),
            ("Net-_U1-+_", "U1+", 282, 139),
            ("Net-_C2-Pad1_", "C2-L", 282, 248),
            ("Net-_U1--_", "U1-", 405, 248),
            ("Net-_U2-+_", "U2+", 585, 350),
            ("/v_filt", "v_filt", 585, 248),
            ("Net-_D1-A_", "D1-A", 702, 248),
            ("Net-_D1-K_", "D1-K", 840, 263),
            ("Net-_U3-+_", "U3+", 685, 300),
            ("/v_env", "v_env", 919, 263),
            ("/v_thr", "v_thr", 1005, 295),
            ("/v_comp", "v_comp", 1140, 276),
            ("0.9V", "0.9V", 585, 454),
            ("1.8V", "1.8V", 500, 76),
            ("GND", "GND", 88, 525),
        )
        for node, display_name, x_pos, y_pos in node_positions:
            voltage = self._lookup_voltage(result.op_voltages, node)
            text = (
                f"{display_name} {format_voltage_mv(voltage)}"
                if voltage is not None
                else f"{display_name} N/A"
            )
            self._canvas_boxed_text(
                canvas,
                x_pos * scale,
                y_pos * scale,
                text,
                "#0d47a1",
                "#e3f2fd",
                scale,
                font_size=8,
            )

        # The editor already shows current AppState values, so the overlay uses
        # them too.  Otherwise an op-amp or component edit made in another tab
        # would stay invisible here until the next run.
        try:
            applied = self._channel_by_number(result.channel.ch)
        except ValueError:
            applied = result.channel
        parameters = applied.spice_parameters()
        for parameter, x_pos, y_pos in SCHEMATIC_COMPONENT_POSITIONS:
            self._canvas_boxed_text(
                canvas,
                x_pos * scale,
                y_pos * scale,
                self._schematic_component_value(parameters[parameter]),
                "#1b5e20",
                "#e8f5e9",
                scale,
                anchor="center",
                font_size=SCHEMATIC_VALUE_FONT_SIZE,
                padding=1,
            )
        # The detector op-amp is selectable, so the template's printed OPA379
        # is covered with whatever model the next run will actually use.
        opamp = applied.detector_opamp
        is_default = opamp == DEFAULT_DETECTOR_OPAMP
        self._canvas_boxed_text(
            canvas,
            SCHEMATIC_DETECTOR_OPAMP_POSITION[0] * scale,
            SCHEMATIC_DETECTOR_OPAMP_POSITION[1] * scale,
            opamp,
            "#1b5e20" if is_default else "#ad1457",
            "#e8f5e9" if is_default else "#fce4ec",
            scale,
            anchor="center",
            font_size=SCHEMATIC_VALUE_FONT_SIZE,
            padding=1,
        )
        canvas.configure(scrollregion=(0, 0, width, height))

    @staticmethod
    def _canvas_boxed_text(
        canvas: object,
        x_pos: float,
        y_pos: float,
        text: str,
        foreground: str,
        background: str,
        scale: float,
        anchor: str = "sw",
        font_size: int = 10,
        padding: int = 2,
    ) -> None:
        """Draw legible foreground text on a color-matched opaque rectangle."""

        text_id = canvas.create_text(
            x_pos,
            y_pos,
            text=text,
            anchor=anchor,
            fill=foreground,
            font=("Segoe UI", max(7, round(font_size * scale)), "bold"),
        )
        bounds = canvas.bbox(text_id)
        if bounds:
            rectangle = canvas.create_rectangle(
                bounds[0] - padding,
                bounds[1] - 1,
                bounds[2] + padding,
                bounds[3] + 1,
                fill=background,
                outline=foreground,
                width=1,
            )
            canvas.tag_lower(rectangle, text_id)

    def _build_op_detail_tables(
        self, parent: object, result: ChannelResult
    ) -> None:
        """Build the all-node mV table and editable next-run component form."""

        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        node_tab = self.ttk.Frame(notebook)
        component_tab = self.ttk.Frame(notebook)
        notebook.add(node_tab, text=f"노드 전압 ({len(result.op_voltages)})")
        notebook.add(component_tab, text="적용 소자값 편집")
        self.op_detail_notebook = notebook
        self.op_detail_tabs = {
            "node": node_tab,
            "components": component_tab,
        }
        selected_detail = self.op_detail_tabs.get(
            self.op_detail_selected_tab,
            node_tab,
        )
        notebook.select(selected_detail)
        notebook.bind(
            "<<NotebookTabChanged>>",
            self._on_op_detail_tab_changed,
        )

        node_table = self.ttk.Treeview(
            node_tab,
            columns=("node", "voltage"),
            show="headings",
        )
        node_table.heading("node", text="노드")
        node_table.heading("voltage", text="전압 [mV]")
        node_table.column("node", width=210, stretch=True)
        node_table.column("voltage", width=115, anchor="e")
        node_scroll = self.ttk.Scrollbar(
            node_tab, orient="vertical", command=node_table.yview
        )
        node_table.configure(yscrollcommand=node_scroll.set)
        node_table.grid(row=0, column=0, sticky="nsew")
        node_scroll.grid(row=0, column=1, sticky="ns")
        node_tab.rowconfigure(0, weight=1)
        node_tab.columnconfigure(0, weight=1)
        for node, voltage in sorted(
            result.op_voltages.items(),
            key=lambda item: _canonical_vector_name(item[0]),
        ):
            node_table.insert(
                "", "end", values=(node, format_millivolts(voltage))
            )

        applied = self._channel_by_number(result.channel.ch)
        self.component_editor_channel = applied.ch
        self.component_edit_vars = {}
        self.component_divider_edit_source = "resistors"
        self.component_divider_nominal_total_kohm = (
            applied.r7_kohm + applied.r8_kohm
        )
        self.component_vdet_v = self._lookup_voltage(
            result.op_voltages,
            "/v_env",
        )
        editor = self.ttk.Frame(component_tab, padding=7)
        editor.pack(fill="both", expand=True)
        self.ttk.Label(
            editor,
            text=(
                "편집값은 메모리에만 적용되며 기본 CSV는 바뀌지 않습니다.\n"
                "메인 시뮬레이션 버튼을 누르면 현재 입력값을 자동 적용합니다."
            ),
            justify="left",
            foreground="#7a3e00",
            wraplength=310,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.ttk.Label(
            editor,
            text=(
                f"ch{applied.ch:02d} · 설계 fc={applied.f_c_hz:g} Hz · "
                f"설계 Q={applied.q:g}"
            ),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self._set_dc_component_change_status(applied.ch)
        self.ttk.Label(
            editor,
            textvariable=self.dc_component_change_state_var,
            foreground="#9a5500",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        for row, (key, label, attribute) in enumerate(
            COMPONENT_EDIT_FIELDS, start=3
        ):
            value = getattr(applied, attribute)
            variable = self.tk.StringVar(
                value=(
                    format_resistance_kohm(value)
                    if key in RESISTANCE_EDIT_KEYS
                    else f"{value:.12g}"
                )
            )
            self.component_edit_vars[key] = variable
            self.ttk.Label(editor, text=label).grid(
                row=row, column=0, sticky="e", padx=(0, 6), pady=2
            )
            entry = self.ttk.Entry(editor, textvariable=variable, width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if key in {"R7", "R8"}:
                entry.bind(
                    "<KeyRelease>",
                    lambda _event: self._preview_vref_from_resistors(),
                )
            if key in {"R4", "R5"}:
                variable.trace_add(
                    "write",
                    lambda *_args: self._preview_r6_from_r4_r5(),
                )
            entry.bind(
                "<Return>",
                lambda _event: self._apply_component_editor_from_gui(),
            )
        opamp_row = 3 + len(COMPONENT_EDIT_FIELDS)
        self.ttk.Label(editor, text="U3 opamp").grid(
            row=opamp_row, column=0, sticky="e", padx=(0, 6), pady=(4, 2)
        )
        opamp_combo = self.ttk.Combobox(
            editor,
            textvariable=self.component_opamp_var,
            values=DETECTOR_OPAMP_NAMES,
            state="readonly",
            width=14,
        )
        opamp_combo.grid(row=opamp_row, column=1, sticky="ew", pady=(4, 2))
        opamp_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._apply_component_editor_from_gui(),
        )
        self.component_opamp_var.set(applied.detector_opamp)

        threshold_row = opamp_row + 1
        self.ttk.Label(editor, text="목표 V_margin [mV]").grid(
            row=threshold_row, column=0, sticky="e", padx=(0, 6), pady=(4, 2)
        )
        margin_entry = self.ttk.Entry(
            editor,
            textvariable=self.component_margin_var,
            width=16,
        )
        margin_entry.grid(
            row=threshold_row,
            column=1,
            sticky="ew",
            pady=(4, 2),
        )
        margin_entry.bind(
            "<KeyRelease>",
            lambda _event: self._preview_resistors_from_margin(),
        )
        margin_entry.bind(
            "<Return>",
            lambda _event: self._apply_component_editor_from_gui(),
        )
        simulated_vref = self._lookup_voltage(result.op_voltages, "/v_thr")
        initial_vref = (
            simulated_vref
            if simulated_vref is not None
            else applied.vthr_v
        )
        if self.component_vdet_v is not None:
            initial_margin_v = voltage_margin_v(
                self.component_vdet_v,
                initial_vref,
            )
            self.component_margin_var.set(
                format_millivolts(initial_margin_v)
            )
            self.component_actual_margin_var.set(
                format_margin_mv(initial_margin_v)
            )
        else:
            self.component_margin_var.set("")
            self.component_actual_margin_var.set("—")

        self.ttk.Label(editor, text="Vthr [mV]").grid(
            row=threshold_row + 1,
            column=0,
            sticky="e",
            padx=(0, 6),
            pady=2,
        )
        vref_entry = self.ttk.Entry(
            editor,
            textvariable=self.component_vref_var,
            width=16,
        )
        vref_entry.bind(
            "<KeyRelease>",
            lambda _event: self._preview_margin_from_vthr(),
        )
        vref_entry.bind(
            "<Return>",
            lambda _event: self._apply_component_editor_from_gui(),
        )
        vref_entry.grid(
            row=threshold_row + 1,
            column=1,
            sticky="ew",
            pady=2,
        )
        visible_vref_v = divider_vref_v(
            float(self.component_edit_vars["R7"].get()),
            float(self.component_edit_vars["R8"].get()),
        )
        self.component_vref_var.set(format_millivolts(visible_vref_v))
        self.ttk.Label(
            editor,
            text=(
                "Vmargin = Vthr - Venv_DC · 둘 중 하나를 입력하면 나머지가 "
                "자동 계산되고 R7/R8이 다시 잡힙니다 · "
                "모든 저항은 0.01 kΩ로 반올림 · "
                "R7+R8 합계 고정 안 함 · "
                "R4/R5 변경 시 R6=R4||R5 자동 입력"
            ),
            foreground="#164a7b",
            justify="left",
            wraplength=310,
        ).grid(
            row=threshold_row + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 3),
        )

        self.ttk.Label(
            editor,
            text="최근 DC 실제 Vmargin (Vthr - Venv_DC)",
        ).grid(
            row=threshold_row + 3,
            column=0,
            sticky="e",
            padx=(0, 6),
            pady=2,
        )
        self.ttk.Label(
            editor,
            textvariable=self.component_actual_margin_var,
        ).grid(row=threshold_row + 3, column=1, sticky="w", pady=2)

        buttons = self.ttk.Frame(editor)
        buttons.grid(
            row=threshold_row + 4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(7, 4),
        )
        self.ttk.Button(
            buttons,
            text="변경값 적용 + DC 실행",
            command=self._apply_component_editor_from_gui,
        ).pack(side="left")
        self.ttk.Button(
            buttons,
            text="새 버전 저장",
            command=self._save_component_version_from_gui,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            buttons,
            text="버전 불러오기",
            command=self._browse_table,
        ).pack(side="left", padx=(5, 0))

        models = self.ttk.LabelFrame(editor, text="고정 소자 / 모델", padding=5)
        models.grid(
            row=threshold_row + 5,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        fixed_rows = (
            ("V1", "DC 0.9 V"),
            ("V2", "DC 0.9 V"),
            ("V3", "DC 0.9 V"),
            ("V4", "DC 0 V / AC 1 V"),
            ("U1, U2", "OPA379"),
            ("U4", "LPV7215"),
            ("D1, D2", "BAT54WT1"),
        )
        self.dc_detector_opamp_var.set(applied.detector_opamp)
        self.ttk.Label(models, text="U3 (detector)", width=12).grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(
            models,
            textvariable=self.dc_detector_opamp_var,
            foreground="#ad1457",
        ).grid(row=0, column=1, sticky="w")
        for row, (reference, value) in enumerate(fixed_rows, start=1):
            self.ttk.Label(models, text=reference, width=12).grid(
                row=row, column=0, sticky="w"
            )
            self.ttk.Label(models, text=value).grid(
                row=row, column=1, sticky="w"
            )
        editor.columnconfigure(1, weight=1)

    def _set_dc_component_change_status(self, channel_number: int) -> None:
        """Show whether this channel was edited here or synchronized externally."""

        if self.component_last_change_channel != channel_number:
            self.dc_component_change_state_var.set("현재 AppState 소자값")
        elif self.component_last_change_source == "dc":
            self.dc_component_change_state_var.set("변경됨")
        else:
            self.dc_component_change_state_var.set("다른 탭에서 변경됨")

    def _sync_dc_component_editor(
        self,
        channel_number: int,
        external: bool,
        changed_channels: set[int] | None = None,
    ) -> None:
        """Refresh DC widget text from AppState without committing another edit."""

        changed = changed_channels or {channel_number}
        if (
            self.component_editor_channel not in changed
            or not self.component_edit_vars
        ):
            return
        channel = self._channel_by_number(self.component_editor_channel)
        self.component_auto_update_suspended = True
        try:
            for key, _label, attribute in COMPONENT_EDIT_FIELDS:
                variable = self.component_edit_vars.get(key)
                if variable is None:
                    continue
                value = getattr(channel, attribute)
                variable.set(
                    format_resistance_kohm(value)
                    if key in RESISTANCE_EDIT_KEYS
                    else f"{value:.12g}"
                )
            self.component_vref_var.set(format_millivolts(channel.vthr_v))
            self.component_opamp_var.set(channel.detector_opamp)
            if self.component_vdet_v is not None:
                self.component_margin_var.set(
                    format_millivolts(
                        voltage_margin_v(self.component_vdet_v, channel.vthr_v)
                    )
                )
        finally:
            self.component_auto_update_suspended = False
        self.dc_component_change_state_var.set(
            "다른 탭에서 변경됨" if external else "변경됨"
        )
        self._refresh_dc_model_summary(channel)

    def _refresh_dc_model_summary(self, channel: Channel) -> None:
        """Update the fixed-model row and redraw the schematic in place.

        The op-amp and component values are drawn from AppState, so an edit made
        in the Detector or Transient tab must repaint them here immediately.
        """

        variable = getattr(self, "dc_detector_opamp_var", None)
        if variable is not None:
            variable.set(channel.detector_opamp)
        canvas = self.schematic_canvas
        result = self.schematic_result
        if canvas is None or result is None:
            return
        if result.channel.ch != channel.ch:
            return
        try:
            if canvas.winfo_exists():
                self._draw_operating_point_schematic(canvas, result)
        except Exception:
            pass

    def _channel_by_number(self, channel_number: int) -> Channel:
        """Return the current in-memory channel used by the next simulation."""

        for channel in self.channels:
            if channel.ch == channel_number:
                return channel
        raise ValueError(f"적용 소자값에 ch{channel_number:02d}가 없습니다.")

    def _on_op_detail_tab_changed(self, event: object = None) -> None:
        """Remember whether the user is viewing node voltages or component edits."""

        event_notebook = getattr(event, "widget", None)
        notebook = event_notebook or self.op_detail_notebook
        if notebook is not self.op_detail_notebook:
            return
        if notebook is None:
            return
        try:
            selected = notebook.select()
        except Exception:
            return
        for name, tab in self.op_detail_tabs.items():
            if str(tab) == selected:
                self.op_detail_selected_tab = name
                return

    def _preview_r6_from_r4_r5(self) -> None:
        """Auto-fill R6=R4||R5 only when R4 or R5 itself changes."""

        if self.component_auto_update_suspended:
            return
        try:
            r4_kohm = float(
                str(self.component_edit_vars["R4"].get()).strip()
            )
            r5_kohm = float(
                str(self.component_edit_vars["R5"].get()).strip()
            )
            r6_kohm = parallel_resistance_kohm(r4_kohm, r5_kohm)
        except (KeyError, TypeError, ValueError):
            return
        self.component_edit_vars["R6"].set(
            format_resistance_kohm(r6_kohm)
        )

    def _format_component_resistance_fields(self, channel: Channel) -> None:
        """Normalize all visible resistor fields without retriggering R6."""

        self.component_auto_update_suspended = True
        try:
            for key, _label, attribute in COMPONENT_EDIT_FIELDS:
                if key not in RESISTANCE_EDIT_KEYS:
                    continue
                variable = self.component_edit_vars.get(key)
                if variable is not None:
                    variable.set(
                        format_resistance_kohm(
                            getattr(channel, attribute)
                        )
                    )
        finally:
            self.component_auto_update_suspended = False

    def _preview_vref_from_resistors(self) -> None:
        """Update read-only Vthr and target margin after R7/R8 is typed."""

        self.component_divider_edit_source = "resistors"
        try:
            r7 = float(str(self.component_edit_vars["R7"].get()).strip())
            r8 = float(str(self.component_edit_vars["R8"].get()).strip())
            vref_v = divider_vref_v(r7, r8)
        except (KeyError, TypeError, ValueError):
            return
        self.component_divider_nominal_total_kohm = r7 + r8
        self.component_vref_var.set(format_millivolts(vref_v))
        if self.component_vdet_v is not None:
            margin_v = voltage_margin_v(self.component_vdet_v, vref_v)
            self.component_margin_var.set(format_millivolts(margin_v))

    def _preview_margin_from_vthr(self) -> None:
        """Derive Vmargin and the 0.01 kOhm divider from a typed Vthr.

        Vthr and Vmargin are two views of the same divider, so editing either
        one updates the other and both end up in the same R7/R8 pair.
        """

        if self.component_auto_update_suspended:
            return
        self.component_divider_edit_source = "vthr"
        try:
            vthr_v = float(self.component_vref_var.get().strip()) / 1000.0
            r7_kohm, r8_kohm = divider_resistors_for_vref(
                vthr_v,
                nominal_total_kohm=(
                    self.component_divider_nominal_total_kohm
                ),
            )
        except (KeyError, TypeError, ValueError):
            return
        self.component_edit_vars["R7"].set(format_resistance_kohm(r7_kohm))
        self.component_edit_vars["R8"].set(format_resistance_kohm(r8_kohm))
        if self.component_vdet_v is not None:
            self.component_margin_var.set(
                format_millivolts(
                    voltage_margin_v(self.component_vdet_v, vthr_v)
                )
            )

    def _preview_resistors_from_margin(self) -> None:
        """Derive Vthr and a 0.01 kOhm divider from the target margin."""

        self.component_divider_edit_source = "margin"
        try:
            if self.component_vdet_v is None:
                return
            margin_v = float(self.component_margin_var.get().strip()) / 1000.0
            vref_v = vref_from_margin_v(self.component_vdet_v, margin_v)
            r7_kohm, r8_kohm = divider_resistors_for_vref(
                vref_v,
                nominal_total_kohm=(
                    self.component_divider_nominal_total_kohm
                ),
            )
            self.component_edit_vars["R7"].set(
                format_resistance_kohm(r7_kohm)
            )
            self.component_edit_vars["R8"].set(
                format_resistance_kohm(r8_kohm)
            )
            rounded_vref_v = divider_vref_v(r7_kohm, r8_kohm)
            self.component_vref_var.set(
                format_millivolts(rounded_vref_v)
            )
        except (KeyError, TypeError, ValueError):
            return

    def _component_editor_has_changes(self) -> bool:
        """Detect unsaved text edits without requiring the text to be valid."""

        if self.component_editor_channel is None or not self.component_edit_vars:
            return False
        channel = self._channel_by_number(self.component_editor_channel)
        for key, _label, attribute in COMPONENT_EDIT_FIELDS:
            variable = self.component_edit_vars.get(key)
            if variable is None:
                return True
            try:
                value = float(str(variable.get()).strip())
            except ValueError:
                return True
            expected = float(getattr(channel, attribute))
            if key in RESISTANCE_EDIT_KEYS:
                expected = round_resistance_kohm(expected)
            if not math.isclose(
                value,
                expected,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                return True
        if self.component_divider_edit_source == "margin":
            try:
                float(self.component_margin_var.get().strip())
            except ValueError:
                return True
        return False

    def _commit_component_editor(self) -> None:
        """Validate the visible component form and update its in-memory channel."""

        if self.component_editor_channel is None or not self.component_edit_vars:
            return
        original = self._channel_by_number(self.component_editor_channel)
        edit_source = self.component_divider_edit_source
        values = {
            key: str(variable.get())
            for key, variable in self.component_edit_vars.items()
        }
        requested_margin_mv = (
            self.component_margin_var.get()
            if edit_source == "margin"
            else None
        )
        requested_vthr_mv = (
            self.component_vref_var.get() if edit_source == "vthr" else None
        )
        updated = apply_component_updates(
            original,
            values,
            requested_margin_mv=requested_margin_mv,
            reference_vdet_v=self.component_vdet_v,
            divider_nominal_total_kohm=(
                self.component_divider_nominal_total_kohm
                if edit_source in {"margin", "vthr"}
                else None
            ),
            requested_vthr_mv=requested_vthr_mv,
            detector_opamp=self.component_opamp_var.get().strip() or None,
        )
        self._format_component_resistance_fields(updated)
        self.component_vref_var.set(format_millivolts(updated.vthr_v))
        self.component_opamp_var.set(updated.detector_opamp)
        if self.component_vdet_v is not None:
            self.component_margin_var.set(
                format_millivolts(
                    voltage_margin_v(self.component_vdet_v, updated.vthr_v)
                )
            )
        if edit_source == "resistors":
            self.component_divider_nominal_total_kohm = (
                updated.r7_kohm + updated.r8_kohm
            )
        self.component_divider_edit_source = "resistors"
        if updated == original:
            return
        self.channels = [
            updated if channel.ch == updated.ch else channel
            for channel in self.channels
        ]
        self.component_revision += 1
        self.component_dirty = True
        base_label = (
            "기본값 (읽기 전용)"
            if self.component_source_path == DEFAULT_TABLE.resolve()
            else self.component_source_path.name
        )
        self.component_source_label_var.set(base_label + " / 메모리 수정")
        self._record_component_change("dc", updated.ch)
        self._refresh_transient_component_state()

    def _apply_component_editor_from_gui(self) -> None:
        """Apply edited values and immediately rerun DC OP for that channel."""

        from tkinter import messagebox

        try:
            self._capture_operating_point_view_state()
            self.op_pending_view_state = (
                self.op_detail_selected_tab,
                self.op_pane_fraction,
                self.op_canvas_view,
            )
            channel_number = self.component_editor_channel
            target_margin_v = None
            if self.component_divider_edit_source == "margin":
                try:
                    target_margin_v = (
                        float(self.component_margin_var.get().strip())
                        / 1000.0
                    )
                except ValueError as exc:
                    raise ValueError(
                        "목표 Vmargin: mV 단위 숫자를 입력하세요."
                    ) from exc
            self._commit_component_editor()
            if channel_number is not None:
                updated = self._channel_by_number(channel_number)
                self.pending_margin_retune = (
                    (channel_number, target_margin_v)
                    if target_margin_v is not None
                    else None
                )
                self.status_var.set(
                    f"ch{channel_number:02d} 변경값 적용됨 / "
                    f"Vthr={format_voltage_mv(updated.vthr_v)} / "
                    "DC 동작점 자동 실행"
                )
                self._start_run(
                    ("op",),
                    selected_channels=(updated,),
                )
        except Exception as exc:
            self.op_pending_view_state = None
            messagebox.showerror("소자값 입력 오류", str(exc))

    def _prepare_margin_retune(
        self,
        results: Sequence[ChannelResult],
    ) -> Channel | None:
        """Use the first new OP Venv_DC to refine target Vmargin once."""

        pending = self.pending_margin_retune
        if pending is None:
            return None
        self.pending_margin_retune = None
        channel_number, target_margin_v = pending
        result = next(
            (
                item
                for item in results
                if item.channel.ch == channel_number and item.op_voltages
            ),
            None,
        )
        if result is None:
            return None
        vdet_v = self._lookup_voltage(
            result.op_voltages,
            "/v_env",
        )
        if vdet_v is None:
            return None
        current = self._channel_by_number(channel_number)
        target_vref_v = vref_from_margin_v(vdet_v, target_margin_v)
        r7_kohm, r8_kohm = divider_resistors_for_vref(
            target_vref_v,
            nominal_total_kohm=(
                self.component_divider_nominal_total_kohm
                if self.component_editor_channel == channel_number
                else current.r7_kohm + current.r8_kohm
            ),
        )
        rounded_vref_v = divider_vref_v(r7_kohm, r8_kohm)
        updated = replace(
            current,
            r7_kohm=r7_kohm,
            r8_kohm=r8_kohm,
            vthr_v=rounded_vref_v,
        )
        if updated == current:
            return None
        self.channels = [
            updated if channel.ch == channel_number else channel
            for channel in self.channels
        ]
        self.component_revision += 1
        self.component_dirty = True
        self._record_component_change("dc", channel_number)
        if (
            self.component_editor_channel == channel_number
            and self.component_edit_vars
        ):
            self._format_component_resistance_fields(updated)
            self.component_vref_var.set(
                format_millivolts(rounded_vref_v)
            )
            self.component_divider_edit_source = "resistors"
        return updated

    def _on_op_channel_changed(self, _event: object) -> None:
        """Commit the previous channel before switching the schematic/editor."""

        from tkinter import messagebox

        previous = self.component_editor_channel
        try:
            self._commit_component_editor()
        except Exception as exc:
            if previous is not None:
                for result in self.last_op_results:
                    if result.channel.ch == previous:
                        self.selected_op_channel_var.set(
                            f"ch{previous:02d} ({result.channel.f_c_hz:g} Hz)"
                        )
                        break
            messagebox.showerror("소자값 입력 오류", str(exc))
            return
        self._refresh_operating_point_tab()

    @staticmethod
    def _lookup_voltage(
        voltages: dict[str, float], requested_node: str
    ) -> float | None:
        """Find a voltage despite KiCad slash and ngspice ``v()`` spelling."""

        wanted = _canonical_vector_name(requested_node)
        if wanted in {"0", "gnd"}:
            return 0.0
        for node, voltage in voltages.items():
            if _canonical_vector_name(node) == wanted:
                return voltage
        return None

    @staticmethod
    def _component_value(spice_text: str) -> str:
        """Convert ngspice k/n suffixes into human-readable schematic units."""

        if spice_text.endswith("k"):
            return spice_text[:-1] + " kΩ"
        if spice_text.endswith("n"):
            return spice_text[:-1] + " nF"
        return spice_text

    @staticmethod
    def _schematic_component_value(spice_text: str) -> str:
        """Return the compact form drawn over a ``{NAME}`` placeholder.

        The placeholders are only about 22 px wide, so the bare SPICE suffix is
        kept instead of ``kΩ``/``nF`` to avoid covering the reference
        designators printed beside them.
        """

        return spice_text

    def _component_rows(self, channel: Channel) -> tuple[tuple[str, str], ...]:
        """Return the full applied component/model summary for one channel."""

        p = channel.spice_parameters()
        return (
            ("R1", self._component_value(p["R1"])),
            ("RA1, RA2", self._component_value(p["RA"])),
            ("R2, R3", self._component_value(p["R2"])),
            ("R4", self._component_value(p["R4"])),
            ("R5", self._component_value(p["R5"])),
            ("R6", self._component_value(p["R6"])),
            ("R7", self._component_value(p["R7"])),
            ("R8", self._component_value(p["R8"])),
            ("C1, C2", self._component_value(p["C1"])),
            ("C3", self._component_value(p["C3"])),
            ("V1", "DC 0.9 V"),
            ("V2", "DC 0.9 V"),
            ("V3", "DC 0.9 V"),
            ("V4", "DC 0 V / AC 1 V"),
            ("U1, U2", "OPA379"),
            ("U3 (detector)", channel.detector_opamp),
            ("U4", "LPV7215"),
            ("D1, D2", "BAT54WT1"),
        )
