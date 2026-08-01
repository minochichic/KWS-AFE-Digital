"""Shared file, worker, result-dispatch, and application controls."""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence

from ..components import (
    apply_component_updates,
    load_channels,
    save_component_version,
)
from ..constants import (
    C3_ALLOWED_NF_LABELS,
    COMPONENT_EDIT_FIELDS,
    DEFAULT_COMPONENT_VERSIONS,
    DEFAULT_DETECTOR_OPAMP,
    DEFAULT_TABLE,
    DETECTOR_OPAMP_NAMES,
    DIVIDER_NOMINAL_TOTAL_KOHM,
)
from ..models import (
    Channel,
    ChannelResult,
    SweepSettings,
    normalize_analyses,
)
from ..simulation import Simulator, find_ngspice
from ..values import (
    format_millivolts,
    format_resistance_kohm,
    parallel_resistance_kohm,
)
from .base import AppController
from .cursors import InteractivePlotCursor


class CommonController(AppController):
    """Shared file, worker, result-dispatch, and application controls."""

    def _build_file_settings(self, parent: object) -> object:
        """Build one view of the shared paths and component-version controls."""

        files = self.ttk.LabelFrame(parent, text="파일 설정", padding=8)
        self._path_row(files, 0, "ngspice", self.ngspice_var, self._browse_ngspice)
        self._path_row(files, 1, "네트리스트", self.netlist_var, self._browse_netlist)
        self._path_row(
            files,
            2,
            "적용 CSV",
            self.table_var,
            self._reset_component_defaults,
            entry_state="readonly",
            button_text="기본값",
        )
        version_controls = self.ttk.Frame(files)
        version_controls.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(4, 2)
        )
        self.ttk.Label(version_controls, text="CSV 버전", width=11).pack(
            side="left"
        )
        self.ttk.Label(
            version_controls,
            textvariable=self.component_source_label_var,
        ).pack(side="left", fill="x", expand=True)
        self.ttk.Button(
            version_controls,
            text="새 버전 저장",
            command=self._save_component_version_from_gui,
        ).pack(side="right")
        self.ttk.Button(
            version_controls,
            text="버전 불러오기",
            command=self._browse_table,
        ).pack(side="right", padx=(4, 5))
        self._path_row(
            files, 4, "결과 폴더", self.output_var, self._browse_output
        )
        files.columnconfigure(1, weight=1)
        return files

    def _build_channel_selector(
        self,
        parent: object,
        attribute_name: str,
        select_all_command: Callable[[], None],
        clear_command: Callable[[], None],
        *,
        title: str = "채널",
        on_select: Callable[[], None] | None = None,
    ) -> object:
        """Build a reusable multi-channel selector and store its listbox.

        ``on_select`` runs after every selection change so a tab that also
        shows per-channel values, such as the shared component panel, can
        follow the clicked channel instead of staying on the first one.
        """

        frame = self.ttk.LabelFrame(parent, text=title, padding=8)
        listbox = self.tk.Listbox(
            frame,
            selectmode="extended",
            exportselection=False,
            height=7,
            width=25,
        )
        listbox.grid(row=0, column=0, columnspan=3, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(
            frame, orient="vertical", command=listbox.yview
        )
        scrollbar.grid(row=0, column=3, sticky="ns")
        listbox.configure(yscrollcommand=scrollbar.set)
        def handle_selection(_event: object) -> None:
            """Refresh the run button and any per-channel view this tab owns."""

            self._update_run_button()
            if on_select is not None:
                on_select()

        listbox.bind("<<ListboxSelect>>", handle_selection)
        self.ttk.Button(
            frame, text="전체 선택", command=select_all_command
        ).grid(row=1, column=0, pady=(5, 0))
        self.ttk.Button(frame, text="해제", command=clear_command).grid(
            row=1, column=1, pady=(5, 0)
        )
        self.ttk.Button(
            frame, text="CSV 다시 읽기", command=self._reload_channels
        ).grid(row=1, column=2, pady=(5, 0))
        self.ttk.Button(
            frame,
            text="소자값 보기",
            command=lambda name=attribute_name: self._show_channel_components(
                name
            ),
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        setattr(self, attribute_name, listbox)
        return frame

    def _build_shared_component_editor(
        self,
        parent: object,
        prefix: str,
        *,
        title: str = "공유 소자값",
        note: str = "",
    ) -> object:
        """Build the R4/R5/R6/C3/Vthr/op-amp editor shared by two tabs.

        Detector and Transient show the same panel, so one builder keeps their
        fields, validation, and commit behaviour identical.  ``prefix`` names
        the per-tab widget-variable dictionary on the application.
        """

        frame = self.ttk.LabelFrame(parent, text=title, padding=7)
        variables: dict[str, object] = {}
        setattr(self, f"{prefix}_component_vars", variables)
        # The panel edits one channel at a time and each tab picks its own, so
        # the frame is kept to show which channel the values belong to.
        setattr(self, f"{prefix}_component_frame", frame)
        setattr(self, f"{prefix}_component_title", title)
        commit = lambda: self._commit_shared_components(prefix)
        for row, (label, key) in enumerate(
            (("R4 [kohm]", "R4"), ("R5 [kohm]", "R5"), ("R6 [kohm]", "R6"))
        ):
            variable = self.tk.StringVar(value="")
            variables[key] = variable
            self.ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky="e", padx=(0, 5), pady=2
            )
            entry = self.ttk.Entry(frame, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            entry.bind("<Return>", lambda _event: commit())
            entry.bind("<FocusOut>", lambda _event: commit())
            if key in {"R4", "R5"}:
                variable.trace_add(
                    "write",
                    lambda *_args, name=prefix: self._preview_shared_r6(name),
                )
        variables["C3"] = self.tk.StringVar(value="")
        self.ttk.Label(frame, text="C3 [nF]").grid(
            row=3, column=0, sticky="e", padx=(0, 5), pady=2
        )
        c3_combo = self.ttk.Combobox(
            frame,
            textvariable=variables["C3"],
            values=C3_ALLOWED_NF_LABELS,
            state="readonly",
            width=10,
        )
        c3_combo.grid(row=3, column=1, sticky="ew", pady=2)
        c3_combo.bind("<<ComboboxSelected>>", lambda _event: commit())

        variables["Vthr"] = self.tk.StringVar(value="")
        self.ttk.Label(frame, text="Vthr [mV]").grid(
            row=4, column=0, sticky="e", padx=(0, 5), pady=2
        )
        vthr_entry = self.ttk.Entry(
            frame, textvariable=variables["Vthr"], width=12
        )
        vthr_entry.grid(row=4, column=1, sticky="ew", pady=2)
        vthr_entry.bind("<Return>", lambda _event: commit())
        vthr_entry.bind("<FocusOut>", lambda _event: commit())

        variables["opamp"] = self.tk.StringVar(value=DEFAULT_DETECTOR_OPAMP)
        self.ttk.Label(frame, text="U3 opamp").grid(
            row=5, column=0, sticky="e", padx=(0, 5), pady=2
        )
        opamp_combo = self.ttk.Combobox(
            frame,
            textvariable=variables["opamp"],
            values=DETECTOR_OPAMP_NAMES,
            state="readonly",
            width=10,
        )
        opamp_combo.grid(row=5, column=1, sticky="ew", pady=2)
        opamp_combo.bind("<<ComboboxSelected>>", lambda _event: commit())
        setattr(self, f"{prefix}_c3_combo", c3_combo)
        setattr(self, f"{prefix}_opamp_combo", opamp_combo)

        # A distinct name from the tab-wide status line, which already uses
        # "{prefix}_component_state_var" on the Transient tab.
        bulk_var = self.tk.BooleanVar(value=False)
        setattr(self, f"{prefix}_bulk_var", bulk_var)
        self.ttk.Checkbutton(
            frame,
            text="선택 채널 일괄 적용 (Vthr 제외)",
            variable=bulk_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

        state_var = self.tk.StringVar(value="현재 AppState 소자값")
        setattr(self, f"{prefix}_panel_state_var", state_var)
        self.ttk.Label(
            frame, textvariable=state_var, foreground="#9a5500"
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))
        if note:
            self.ttk.Label(
                frame,
                text=note,
                justify="left",
                wraplength=250,
                foreground="#164a7b",
            ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))
        frame.columnconfigure(1, weight=1)
        return frame

    def _shared_component_channel(self, prefix: str) -> int | None:
        """Return the channel number whose values the panel is showing."""

        return getattr(self, f"{prefix}_editing_channel", None)

    def _set_shared_component_title(
        self, prefix: str, channel_number: int | None
    ) -> None:
        """Name the channel a shared panel is editing right now.

        Each tab picks its own channel, so without this the panels can show
        different channels and an edit elsewhere looks like a broken sync.
        """

        frame = getattr(self, f"{prefix}_component_frame", None)
        if frame is None:
            return
        base = getattr(self, f"{prefix}_component_title", "공유 소자값")
        suffix = "" if channel_number is None else f" · ch{channel_number:02d}"
        try:
            frame.configure(text=f"{base}{suffix}")
        except Exception:
            pass

    def _load_shared_component_fields(
        self, prefix: str, channel_number: int
    ) -> None:
        """Load one channel's shared values into a panel without committing."""

        variables = getattr(self, f"{prefix}_component_vars", None)
        if not variables:
            return
        channel = self._channel_by_number(channel_number)
        setattr(self, f"{prefix}_editing_channel", channel_number)
        self._set_shared_component_title(prefix, channel_number)
        setattr(self, f"{prefix}_component_sync_suspended", True)
        try:
            variables["R4"].set(format_resistance_kohm(channel.r4_kohm))
            variables["R5"].set(format_resistance_kohm(channel.r5_kohm))
            variables["R6"].set(format_resistance_kohm(channel.r6_kohm))
            variables["C3"].set(
                "1.0"
                if math.isclose(channel.c3_nf, 1.0)
                else f"{channel.c3_nf:g}"
            )
            variables["Vthr"].set(format_millivolts(channel.vthr_v))
            variables["opamp"].set(channel.detector_opamp)
        finally:
            setattr(self, f"{prefix}_component_sync_suspended", False)
        getattr(self, f"{prefix}_panel_state_var").set(
            "현재 AppState 소자값"
        )

    def _preview_shared_r6(self, prefix: str) -> None:
        """Recalculate two-decimal R6 after the user changes R4 or R5."""

        if getattr(self, f"{prefix}_component_sync_suspended", False):
            return
        variables = getattr(self, f"{prefix}_component_vars", None)
        if not variables:
            return
        try:
            r6_kohm = parallel_resistance_kohm(
                float(variables["R4"].get()), float(variables["R5"].get())
            )
        except (KeyError, TypeError, ValueError):
            return
        setattr(self, f"{prefix}_component_sync_suspended", True)
        try:
            variables["R6"].set(format_resistance_kohm(r6_kohm))
        finally:
            setattr(self, f"{prefix}_component_sync_suspended", False)

    def _shared_commit_targets(self, prefix: str, selected: int) -> list[int]:
        """Return the channels one shared-panel commit should write to.

        With bulk apply off this is just the channel the panel shows.  With it
        on, every channel ticked in that tab's list is included, because
        R4/R5/R6/C3 and the op-amp are normally identical across channels.
        """

        variable = getattr(self, f"{prefix}_bulk_var", None)
        if variable is None or not variable.get():
            return [selected]
        ticked = [
            channel.ch
            for channel in self._channels_from_listbox(
                f"{prefix}_channel_list"
            )
        ]
        if selected not in ticked:
            ticked.append(selected)
        # The panel's own channel goes first so it owns the Vthr edit.
        return [selected, *(number for number in ticked if number != selected)]

    def _commit_shared_components(
        self, prefix: str, channel_number: int | None = None
    ) -> bool:
        """Validate one shared-panel edit and bump the revision at most once."""

        from tkinter import messagebox

        if getattr(self, f"{prefix}_component_sync_suspended", False):
            return False
        variables = getattr(self, f"{prefix}_component_vars", None)
        if not variables:
            return False
        selected = (
            channel_number
            if channel_number is not None
            else self._shared_component_channel(prefix)
        )
        if selected is None:
            return False
        targets = self._shared_commit_targets(prefix, selected)
        try:
            updates: dict[int, Channel] = {}
            for number in targets:
                original = self._channel_by_number(number)
                values = {
                    key: str(getattr(original, attribute))
                    for key, _label, attribute in COMPONENT_EDIT_FIELDS
                }
                for key in ("R4", "R5", "R6"):
                    values[key] = str(variables[key].get())
                values["C3"] = str(variables["C3"].get())
                updated = apply_component_updates(
                    original,
                    values,
                    # Vthr is per-channel tuning, so a bulk apply must not
                    # overwrite the other channels' individual dividers.
                    requested_vthr_mv=(
                        str(variables["Vthr"].get())
                        if number == selected
                        else None
                    ),
                    divider_nominal_total_kohm=(
                        original.r7_kohm + original.r8_kohm
                    ),
                    detector_opamp=str(variables["opamp"].get()),
                )
                if updated != original:
                    updates[number] = updated
        except Exception as exc:
            messagebox.showerror("공유 소자값 오류", str(exc))
            self._load_shared_component_fields(prefix, selected)
            return False
        if not updates:
            return False
        self.channels = [
            updates.get(channel.ch, channel) for channel in self.channels
        ]
        # One user edit counts as one revision even when it touched many
        # channels, so cached results are not invalidated repeatedly.
        self.component_revision += 1
        self.component_dirty = True
        base_label = (
            "기본값 (읽기 전용)"
            if self.component_source_path == DEFAULT_TABLE.resolve()
            else self.component_source_path.name
        )
        self.component_source_label_var.set(base_label + " / 메모리 수정")
        self._load_shared_component_fields(prefix, selected)
        self._record_component_change(
            prefix, selected, changed_channels=set(updates)
        )
        if len(updates) > 1:
            getattr(self, f"{prefix}_panel_state_var").set(
                f"{len(updates)}개 채널에 일괄 적용됨 "
                + ", ".join(f"ch{number:02d}" for number in sorted(updates))
            )
        return True

    def _sync_shared_component_editor(
        self,
        prefix: str,
        channel_number: int,
        external: bool,
        changed_channels: set[int] | None = None,
    ) -> None:
        """Mirror an AppState edit made in another tab into this panel."""

        state_var = getattr(self, f"{prefix}_panel_state_var", None)
        if state_var is None:
            return
        changed = changed_channels or {channel_number}
        showing = self._shared_component_channel(prefix)
        if showing not in changed:
            # Loading the edited channel here would silently switch which
            # channel this tab edits, so only the reason is reported.
            if showing is not None:
                state_var.set(
                    f"ch{channel_number:02d}가 다른 탭에서 변경됨 "
                    f"(이 패널은 ch{showing:02d})"
                )
            return
        self._load_shared_component_fields(prefix, showing)
        state_var.set("다른 탭에서 변경됨" if external else "변경됨")

    def _channels_from_listbox(self, attribute_name: str) -> list[Channel]:
        """Return the channels ticked in one tab's selector listbox."""

        listbox = getattr(self, attribute_name, None)
        if listbox is None:
            return []
        return [
            self.channels[index]
            for index in listbox.curselection()
            if 0 <= index < len(self.channels)
        ]

    def _show_channel_components(self, attribute_name: str) -> None:
        """Open a comparison window of the selected channels' component values.

        The same rows the DC tab already documents are reused, so the window
        always matches what the next simulation will apply.
        """

        from tkinter import messagebox

        selected = self._channels_from_listbox(attribute_name)
        if not selected:
            messagebox.showinfo(
                "소자값 보기",
                "값을 확인할 채널을 목록에서 하나 이상 선택하세요.",
            )
            return
        window = self.tk.Toplevel(self.root)
        window.title(
            "적용 소자값 · "
            + ", ".join(f"ch{channel.ch:02d}" for channel in selected)
        )
        window.geometry("760x560")
        self.ttk.Label(
            window,
            text=(
                "현재 AppState 값입니다. 다음 실행에 그대로 적용됩니다. "
                f"(component_revision {self.component_revision})"
            ),
            foreground="#164a7b",
            padding=(10, 8),
        ).pack(fill="x")
        # 16 channels do not fit on screen, so the table lives in its own frame
        # with both scrollbars and keeps fixed column widths.  The Treeview must
        # be a child of that frame; gridding a Toplevel child into it leaves the
        # table unmapped and the window looks empty.
        host = self.ttk.Frame(window)
        host.pack(fill="both", expand=True, padx=10, pady=10)
        columns = tuple(f"ch{channel.ch:02d}" for channel in selected)
        table = self.ttk.Treeview(
            host,
            columns=("item", *columns),
            show="headings",
        )
        table.heading("item", text="소자")
        table.column("item", width=150, anchor="w", stretch=False)
        for column in columns:
            table.heading(column, text=column)
            table.column(column, width=110, anchor="center", stretch=False)
        rows_by_channel = [
            dict(self._component_rows(channel)) for channel in selected
        ]
        order = [label for label, _value in self._component_rows(selected[0])]
        for label in order:
            table.insert(
                "",
                "end",
                values=(
                    label,
                    *(rows.get(label, "—") for rows in rows_by_channel),
                ),
            )
        for label, getter in (
            ("설계 fc [Hz]", lambda item: f"{item.f_c_hz:g}"),
            ("설계 Q", lambda item: f"{item.q:g}"),
            ("Vthr [mV]", lambda item: format_millivolts(item.vthr_v)),
        ):
            table.insert(
                "",
                "end",
                values=(label, *(getter(channel) for channel in selected)),
            )
        vertical = self.ttk.Scrollbar(
            host, orient="vertical", command=table.yview
        )
        horizontal = self.ttk.Scrollbar(
            host, orient="horizontal", command=table.xview
        )
        table.configure(
            yscrollcommand=vertical.set, xscrollcommand=horizontal.set
        )
        table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)

    def _path_row(
        self,
        parent: object,
        row: int,
        label: str,
        variable: object,
        command: Callable[[], None],
        entry_state: str = "normal",
        button_text: str = "찾기...",
    ) -> None:
        """Add a consistent label/entry/action row to the file-settings panel."""

        self.ttk.Label(parent, text=label, width=11).grid(
            row=row, column=0, sticky="e", padx=(0, 5), pady=2
        )
        self.ttk.Entry(
            parent, textvariable=variable, state=entry_state
        ).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        self.ttk.Button(parent, text=button_text, command=command).grid(
            row=row, column=2, padx=(5, 0), pady=2
        )

    def _browse_ngspice(self) -> None:
        """Let the user choose the Windows ngspice executable."""

        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="ngspice 실행 파일 선택",
            filetypes=(("실행 파일", "*.exe"), ("모든 파일", "*.*")),
        )
        if path:
            self.ngspice_var.set(path)

    def _browse_netlist(self) -> None:
        """Let the user choose the shared circuit netlist template."""

        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="공통 네트리스트 선택",
            filetypes=(("SPICE netlist", "*.cir *.sp *.spice *.net *.txt"), ("모든 파일", "*.*")),
        )
        if path:
            self.netlist_var.set(path)

    def _browse_table(self) -> None:
        """Load a previously saved component version without changing defaults."""

        from tkinter import filedialog, messagebox

        if (
            self.component_dirty or self._component_editor_has_changes()
        ) and not messagebox.askyesno(
            "버전 불러오기",
            "저장하지 않은 소자값 변경을 버리고 다른 버전을 불러올까요?",
        ):
            return

        path = filedialog.askopenfilename(
            title="저장된 채널 소자 버전 불러오기",
            initialdir=str(DEFAULT_COMPONENT_VERSIONS),
            filetypes=(("CSV", "*.csv"), ("모든 파일", "*.*")),
        )
        if path:
            self._reload_channels(Path(path))

    def _browse_output(self) -> None:
        """Let the user choose the parent directory for simulation sessions."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="결과 폴더 선택")
        if path:
            self.output_var.set(path)

    @staticmethod
    def _arm_plot_cursors(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Arm one placement on the next left click across a cursor group."""

        for cursor in cursors:
            cursor.arm()

    @staticmethod
    def _clear_plot_cursors(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Remove all fixed cursors from every axes in a cursor group."""

        for cursor in cursors:
            cursor.clear()

    @staticmethod
    def _dispose_cursor_group(
        cursors: Sequence[InteractivePlotCursor],
    ) -> None:
        """Disconnect all callbacks before destroying a plot's Tk widgets."""

        for cursor in tuple(cursors):
            cursor.dispose()

    def _select_all(self) -> None:
        """Select every loaded channel in the DC run list."""

        self.channel_list.selection_set(0, "end")

    def _clear_selection(self) -> None:
        """Clear the DC run-list channel selection."""

        self.channel_list.selection_clear(0, "end")

    def _reset_component_defaults(self) -> None:
        """Restore the immutable factory CSV into the in-memory applied set."""

        from tkinter import messagebox

        if (
            self.component_dirty or self._component_editor_has_changes()
        ) and not messagebox.askyesno(
            "기본값 복원",
            "저장하지 않은 소자값 변경을 버리고 기본값을 다시 불러올까요?",
        ):
            return
        self._reload_channels(DEFAULT_TABLE)

    def _record_component_change(
        self,
        source: str,
        channel_number: int,
        changed_channels: set[int] | None = None,
    ) -> None:
        """Synchronize component widgets after one already-counted AppState edit.

        ``changed_channels`` lets a bulk apply refresh every panel that shows
        any touched channel, not only the one the edit was typed into.
        """

        changed = changed_channels or {channel_number}
        self.component_last_change_source = source
        self.component_last_change_channel = channel_number
        self._sync_dc_component_editor(
            channel_number,
            external=source != "dc",
            changed_channels=changed,
        )
        for prefix in ("detector", "transient"):
            self._sync_shared_component_editor(
                prefix,
                channel_number,
                external=source != prefix,
                changed_channels=changed,
            )
        self._refresh_transient_component_state()
        self._refresh_detector_component_state()

    def _save_component_version_from_gui(self) -> None:
        """Commit the visible editor and save all 16 channels to a new CSV."""

        from tkinter import messagebox

        try:
            self._commit_component_editor()
            target = save_component_version(self.channels)
            self.component_source_path = target
            self.table_var.set(str(target))
            self.component_source_label_var.set(target.name)
            self.component_dirty = False
            self.status_var.set(f"소자값 버전 저장됨: {target.name}")
            messagebox.showinfo(
                "소자값 버전 저장",
                "기본 channel_components.csv는 변경하지 않았습니다.\n\n"
                f"새 버전:\n{target}",
            )
        except Exception as exc:
            messagebox.showerror("소자값 버전 저장 오류", str(exc))

    def _reload_channels(self, path: Path | None = None) -> None:
        """Install a factory/version CSV as the current in-memory component set."""

        from tkinter import messagebox

        try:
            source = (path or self.component_source_path).expanduser().resolve()
            loaded = load_channels(source)
            previous_channels = tuple(self.channels)
            if source == DEFAULT_TABLE.resolve():
                self.default_channels = list(loaded)
            elif self.default_channels:
                expected = {channel.ch for channel in self.default_channels}
                actual = {channel.ch for channel in loaded}
                if actual != expected:
                    raise ValueError(
                        "버전 CSV의 채널 집합이 기본 CSV와 다릅니다. "
                        f"기본={sorted(expected)}, 버전={sorted(actual)}"
                    )
            previous_by_list: dict[str, set[int]] = {}
            for attribute in ("channel_list", "ac_channel_list"):
                listbox = getattr(self, attribute, None)
                previous_by_list[attribute] = (
                    {
                        self.channels[index].ch
                        for index in listbox.curselection()
                        if 0 <= index < len(self.channels)
                    }
                    if listbox is not None
                    else set()
                )
            self.channels = loaded
            if tuple(loaded) != previous_channels:
                self.component_revision += 1
                self.component_last_change_source = "csv"
                self.component_last_change_channel = None
            self.component_source_path = source
            self.table_var.set(str(source))
            self.component_source_label_var.set(
                "기본값 (읽기 전용)"
                if source == DEFAULT_TABLE.resolve()
                else source.name
            )
            self.component_dirty = False
            self.component_editor_channel = None
            self.component_edit_vars = {}
            self.component_divider_edit_source = "resistors"
            self.component_divider_nominal_total_kohm = (
                DIVIDER_NOMINAL_TOTAL_KOHM
            )
            self.component_vdet_v = None
            self.pending_margin_retune = None
            self.op_pending_view_state = None
            for attribute in ("channel_list", "ac_channel_list"):
                listbox = getattr(self, attribute, None)
                if listbox is None:
                    continue
                listbox.delete(0, "end")
                for channel in self.channels:
                    listbox.insert(
                        "end", f"ch{channel.ch:02d}   fc={channel.f_c_hz:g} Hz"
                    )
                restored = False
                for index, channel in enumerate(self.channels):
                    if channel.ch in previous_by_list[attribute]:
                        listbox.selection_set(index)
                        restored = True
                if self.channels and not restored:
                    listbox.selection_set(0)
            self.status_var.set(
                f"{len(self.channels)}개 채널 준비됨 / "
                f"{self.component_source_label_var.get()}"
            )
            self._populate_transient_channels()
            self._populate_detector_channels()
            self._refresh_transient_component_state()
            self._refresh_detector_component_state()
            self._update_run_button()
            if self.last_op_results:
                self._refresh_operating_point_tab()
        except Exception as exc:
            messagebox.showerror("소자 테이블 오류", str(exc))
            self.status_var.set("소자 테이블 오류")

    def _selected_analyses(self) -> tuple[str, ...]:
        """Translate the two run checkboxes into analysis job keys."""

        selected: list[str] = []
        if self.run_ac_var.get():
            selected.append("ac")
        if self.run_op_var.get():
            selected.append("op")
        if not selected:
            raise ValueError("AC Sweep 또는 DC 동작점을 하나 이상 체크하세요.")
        return tuple(selected)

    def _update_run_button(self) -> None:
        """Enable the separate DC and AC buttons when their channel lists are ready."""

        running = bool(self.worker and self.worker.is_alive()) or bool(
            self.transient_worker and self.transient_worker.is_alive()
        ) or bool(
            getattr(self, "detector_worker", None)
            and self.detector_worker.is_alive()
        )
        if hasattr(self, "op_run_button"):
            has_dc = bool(self.channel_list.curselection())
            self.op_run_button.configure(
                state="disabled" if running or not has_dc else "normal"
            )
        if hasattr(self, "ac_run_button"):
            has_ac = bool(self.ac_channel_list.curselection())
            self.ac_run_button.configure(
                state="disabled" if running or not has_ac else "normal"
            )

    def _settings(self, analyses: Sequence[str]) -> SweepSettings:
        """Parse current GUI text into a validated settings value object."""

        selected = normalize_analyses(analyses)
        defaults = SweepSettings()
        return SweepSettings(
            ac_points_per_decade=(
                int(self.ac_points_var.get())
                if "ac" in selected
                else defaults.ac_points_per_decade
            ),
            ac_start_hz=(
                float(self.ac_start_var.get())
                if "ac" in selected
                else defaults.ac_start_hz
            ),
            ac_stop_hz=(
                float(self.ac_stop_var.get())
                if "ac" in selected
                else defaults.ac_stop_hz
            ),
            output_node=self.output_node_var.get().strip(),
            pspice_compat=self.pspice_compat_var.get(),
        )

    def _start_run(
        self,
        analyses: Sequence[str] | None = None,
        channel_list: object | None = None,
        selected_channels: Sequence[Channel] | None = None,
    ) -> None:
        """Commit edited components and launch one DC or AC worker job set."""

        from tkinter import messagebox

        if self.worker and self.worker.is_alive():
            return
        try:
            if self.transient_worker and self.transient_worker.is_alive():
                raise RuntimeError("Transient 실행이 끝난 뒤 AC/OP를 시작하세요.")
            if (
                getattr(self, "detector_worker", None)
                and self.detector_worker.is_alive()
            ):
                raise RuntimeError("Active Detector 실행이 끝난 뒤 AC/OP를 시작하세요.")
            self._commit_component_editor()
            requested_analyses = normalize_analyses(
                analyses if analyses is not None else self._selected_analyses()
            )
            if selected_channels is not None:
                selected = list(selected_channels)
            else:
                active_list = channel_list or (
                    self.ac_channel_list
                    if requested_analyses == ("ac",)
                    else self.channel_list
                )
                indices = list(active_list.curselection())
                if not indices:
                    raise ValueError("실행할 채널을 하나 이상 선택하세요.")
                selected = [self.channels[index] for index in indices]
            if not selected:
                raise ValueError("실행할 채널을 하나 이상 선택하세요.")
            ngspice = find_ngspice(self.ngspice_var.get())
            settings = self._settings(requested_analyses)
            settings.validate(requested_analyses)
            netlist = Path(self.netlist_var.get()).expanduser().resolve()
            if not netlist.is_file():
                raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {netlist}")
            output = Path(self.output_var.get()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("실행 설정 오류", str(exc))
            return
        self.log_text.delete("1.0", "end")
        self.simulator = Simulator(
            lambda text: self.events.put(("log", text))
        )
        self.op_run_button.configure(state="disabled")
        self.ac_run_button.configure(state="disabled")
        self.transient_run_button.configure(state="disabled")
        self.detector_run_button.configure(state="disabled")
        self.op_stop_button.configure(state="normal")
        self.ac_stop_button.configure(state="normal")
        if "op" in requested_analyses:
            self.op_progress.start(12)
        if "ac" in requested_analyses:
            self.ac_progress.start(12)
        analysis_label = "DC 동작점" if requested_analyses == ("op",) else "AC"
        self.status_var.set(
            f"{analysis_label} · {len(selected)}개 채널 실행 중..."
        )

        def worker() -> None:
            """Run blocking simulation work outside Tk's event thread."""

            try:
                results = self.simulator.run(
                    selected,
                    netlist,
                    output,
                    ngspice,
                    settings,
                    requested_analyses,
                )
                self.events.put(("done", (requested_analyses, results)))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop_run(self) -> None:
        """Forward the stop request to the active simulator."""

        if self.simulator:
            self.simulator.stop()
            self.status_var.set("중지 요청 중...")

    def _poll_events(self) -> None:
        """Drain worker events on Tk's main thread and update the GUI safely."""

        from tkinter import messagebox

        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log_text.insert("end", str(payload) + "\n")
                    self.log_text.see("end")
                elif kind == "done":
                    requested_analyses, completed = payload  # type: ignore[misc]
                    completed_results = list(completed)
                    retune_channel = None
                    if "op" in requested_analyses and completed_results:
                        try:
                            retune_channel = self._prepare_margin_retune(
                                completed_results
                            )
                        except Exception as exc:
                            self.pending_margin_retune = None
                            messagebox.showerror(
                                "Vmargin 자동 보정 오류",
                                "첫 DC 결과는 표시하지만 새 Venv_DC 기준 "
                                f"R7/R8 보정은 적용하지 못했습니다.\n\n{exc}",
                            )
                    if retune_channel is not None:
                        self._finish_controls()
                        self._start_run(
                            ("op",),
                            selected_channels=(retune_channel,),
                        )
                        self.status_var.set(
                            f"ch{retune_channel.ch:02d} 새 Venv_DC 기준 "
                            "Vmargin 보정 DC 재실행 중..."
                        )
                        continue
                    self.last_results = completed_results
                    if "ac" in requested_analyses:
                        self.last_ac_results = [
                            result for result in completed_results if result.ac_rows
                        ]
                    if "op" in requested_analyses:
                        self.last_op_results = [
                            result
                            for result in completed_results
                            if result.op_voltages
                        ]
                    self._finish_controls()
                    if completed_results:
                        try:
                            has_ac = "ac" in requested_analyses
                            has_op = "op" in requested_analyses
                            self._render_results(
                                completed_results,
                                analyses=requested_analyses,
                            )
                            self.status_var.set(
                                f"{len(completed_results)}개 채널 완료"
                                + (" / AC" if has_ac else "")
                                + (" / OP" if has_op else "")
                            )
                        except Exception as exc:
                            self.status_var.set("해석 완료 / 그래프 표시 실패")
                            self.log_text.insert("end", f"\n그래프 오류: {exc}\n")
                            messagebox.showwarning(
                                "그래프 표시 오류",
                                f"ngspice 해석과 CSV 저장은 완료됐지만 "
                                f"그래프를 표시하지 못했습니다.\n\n{exc}",
                            )
                    else:
                        self.pending_margin_retune = None
                        self.op_pending_view_state = None
                        self.status_var.set("실행 중지됨")
                elif kind == "error":
                    self.pending_margin_retune = None
                    self.op_pending_view_state = None
                    self._finish_controls()
                    self.status_var.set("실행 실패")
                    self.log_text.insert("end", f"\nERROR: {payload}\n")
                    self.log_text.see("end")
                    messagebox.showerror("ngspice 실행 오류", str(payload))
                elif kind == "pwl_progress":
                    position, total, source, target = payload  # type: ignore[misc]
                    self.pwl_progress.configure(maximum=total, value=position - 1)
                    self.pwl_log_text.insert(
                        "end",
                        f"[{position}/{total}] {source} -> {target}\n",
                    )
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set(
                        f"PWL 변환 중: {position}/{total} · {source.name}"
                    )
                elif kind == "pwl_done":
                    results = payload  # type: ignore[assignment]
                    self._finish_pwl_controls()
                    self.pwl_progress.configure(value=len(results))
                    output_root = Path(
                        self.pwl_output_var.get()
                    ).expanduser().resolve()
                    self.pwl_log_text.insert(
                        "end",
                        f"\n완료: {len(results)}개\n"
                        f"manifest: {output_root / 'pwl_manifest.csv'}\n",
                    )
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set(
                        f"PWL {len(results)}개 변환 완료 · "
                        "DC=0 V / 10 mVpp / tail=0 V"
                    )
                    self.transient_folder_var.set(str(output_root))
                    self._refresh_transient_pwl_files()
                elif kind == "pwl_stopped":
                    self._finish_pwl_controls()
                    self.pwl_log_text.insert("end", f"\n중지됨: {payload}\n")
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set("PWL 변환 중지됨")
                elif kind == "pwl_error":
                    self._finish_pwl_controls()
                    self.pwl_log_text.insert("end", f"\nERROR: {payload}\n")
                    self.pwl_log_text.see("end")
                    self.pwl_status_var.set("PWL 변환 실패")
                    messagebox.showerror("PWL 변환 오류", str(payload))
                elif kind == "tran_log":
                    self.transient_log_text.insert(
                        "end", str(payload) + "\n"
                    )
                    self.transient_log_text.see("end")
                elif kind == "tran_result":
                    result = payload  # type: ignore[assignment]
                    self.transient_completed_jobs += 1
                    self._install_transient_result(result)
                    self.transient_status_var.set(
                        f"Transient {self.transient_completed_jobs}/"
                        f"{self.transient_total_jobs} jobs 완료 · "
                        f"ch{result.channel.ch:02d}까지 결과 확인 가능"
                    )
                elif kind == "tran_done":
                    results = payload  # type: ignore[assignment]
                    self._finish_transient_controls()
                    self.last_transient_results = list(results)
                    if self.last_transient_results:
                        self._install_transient_results(
                            self.last_transient_results
                        )
                        self.transient_status_var.set(
                            f"Transient {len(results)} jobs 완료"
                        )
                    else:
                        self.transient_status_var.set("Transient 실행 중지됨")
                elif kind == "tran_error":
                    self._finish_transient_controls()
                    self.transient_log_text.insert(
                        "end", f"\nERROR: {payload}\n"
                    )
                    self.transient_log_text.see("end")
                    self.transient_status_var.set("Transient 실행 실패")
                    messagebox.showerror("Transient 실행 오류", str(payload))
                elif kind == "detector_log":
                    self.detector_log_text.insert("end", str(payload) + "\n")
                    self.detector_log_text.see("end")
                elif kind == "detector_result":
                    result = payload  # type: ignore[assignment]
                    self.detector_completed_jobs += 1
                    self._install_detector_result(result)
                    self.detector_status_var.set(
                        f"Active Detector {self.detector_completed_jobs}/"
                        f"{self.detector_total_jobs} jobs 완료 · "
                        f"ch{result.channel.ch:02d}까지 결과 확인 가능"
                    )
                elif kind == "detector_done":
                    results = payload  # type: ignore[assignment]
                    self._finish_detector_controls()
                    if results:
                        self.detector_status_var.set(
                            f"Active Detector {len(results)} jobs 완료"
                        )
                    else:
                        self.detector_status_var.set(
                            "Active Detector 실행 중지됨"
                        )
                elif kind == "detector_error":
                    self._finish_detector_controls()
                    self.detector_log_text.insert(
                        "end", f"\nERROR: {payload}\n"
                    )
                    self.detector_log_text.see("end")
                    self.detector_status_var.set("Active Detector 실행 실패")
                    messagebox.showerror("Active Detector 실행 오류", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish_controls(self) -> None:
        """Restore controls after success, cancellation, or an error."""

        self.op_progress.stop()
        self.ac_progress.stop()
        self.worker = None
        self.op_stop_button.configure(state="disabled")
        self.ac_stop_button.configure(state="disabled")
        self._update_run_button()
        self._update_transient_run_button()
        self._update_detector_run_button()

    def _show_empty_results(self) -> None:
        """Populate result tabs with instructions before the first simulation."""

        messages = (
            (self.mag_frame, "AC Sweep 실행 후 Magnitude 그래프가 표시됩니다."),
            (self.phase_frame, "AC Sweep 실행 후 Phase 그래프가 표시됩니다."),
            (
                self.op_frame,
                "DC 동작점 (.op) 실행 후 채널별 노드 전압 회로도가 표시됩니다.",
            ),
        )
        for frame, message in messages:
            self.ttk.Label(frame, text=message, justify="center").pack(expand=True)

    @staticmethod
    def _clear_frame(frame: object) -> None:
        """Destroy every child widget in a result-tab frame."""

        for widget in frame.winfo_children():
            widget.destroy()

    def _render_results(
        self,
        results: Sequence[ChannelResult],
        analyses: Sequence[str] | None = None,
    ) -> None:
        """Redraw only the completed analysis while preserving every visible tab."""

        ac_results = [result for result in results if result.ac_rows]
        op_results = [result for result in results if result.op_voltages]
        requested = (
            normalize_analyses(analyses)
            if analyses is not None
            else normalize_analyses(
                (
                    *(("ac",) if ac_results else ()),
                    *(("op",) if op_results else ()),
                )
            )
        )
        selected_ac_tab = self.ac_result_notebook.select()
        if "ac" in requested:
            for group in self.ac_plot_cursors.values():
                self._dispose_cursor_group(group)
                group.clear()
            self.metric_selection_overlay = None
            self.metric_table = None
            for frame in (self.mag_frame, self.phase_frame):
                self._clear_frame(frame)
            if ac_results:
                self._render_magnitude(ac_results)
                self._render_phase(ac_results)
            else:
                self.ttk.Label(
                    self.mag_frame, text="이번 실행에는 AC 결과가 없습니다."
                ).pack(expand=True)
                self.ttk.Label(
                    self.phase_frame, text="이번 실행에는 AC 결과가 없습니다."
                ).pack(expand=True)
            if selected_ac_tab:
                self.ac_result_notebook.select(selected_ac_tab)
        if "op" in requested:
            pending_view = getattr(self, "op_pending_view_state", None)
            if pending_view is None:
                self._capture_operating_point_view_state()
            else:
                (
                    self.op_detail_selected_tab,
                    self.op_pane_fraction,
                    self.op_canvas_view,
                ) = pending_view
            self._clear_frame(self.op_frame)
            if op_results:
                self._render_operating_point(op_results)
            else:
                self.ttk.Label(
                    self.op_frame, text="이번 실행에는 DC 동작점 결과가 없습니다."
                ).pack(expand=True)
            self.op_pending_view_state = None

    def _open_results(self) -> None:
        """Open the most recent session folder in the platform file manager."""

        from tkinter import messagebox

        target = (
            self.last_results[0].run_dir.parent
            if self.last_results
            else Path(self.output_var.get()).expanduser()
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))
