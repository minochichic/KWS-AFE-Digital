"""Transient job selection, execution, result caching, and plotting tab."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Sequence

from wav_pwl import discover_pwl_files

from ..models import TransientResult, TransientSettings
from ..simulation import TransientSimulator, find_ngspice
from ..values import (
    data_occupancy_limits,
    parse_time_seconds,
    recommended_maximum_step_s,
    transient_rows_to_ms_mv,
)
from .base import AppController, configure_plot_fonts
from .cursors import InteractivePlotCursor


class TransientTabController(AppController):
    """Transient job selection, execution, result caching, and plotting tab."""

    def _build_transient_tab(self) -> None:
        """Build transient input selection, timing controls, and result panes."""

        outer = self.ttk.Frame(self.transient_page, padding=10)
        outer.pack(fill="both", expand=True)
        controls = self.ttk.Frame(outer)
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=5)
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(2, weight=2)
        controls.columnconfigure(3, weight=2)

        files = self.ttk.LabelFrame(controls, text="PWL 입력 파일", padding=7)
        files.grid(row=0, column=0, sticky="nsew")
        path_row = self.ttk.Frame(files)
        path_row.pack(fill="x")
        self.ttk.Entry(
            path_row, textvariable=self.transient_folder_var
        ).pack(side="left", fill="x", expand=True)
        self.ttk.Button(
            path_row,
            text="폴더...",
            command=self._browse_transient_folder,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            path_row,
            text="다시 읽기",
            command=self._refresh_transient_pwl_files,
        ).pack(side="left", padx=(5, 0))
        file_list_host = self.ttk.Frame(files)
        file_list_host.pack(fill="both", expand=True, pady=(5, 0))
        self.transient_file_list = self.tk.Listbox(
            file_list_host,
            selectmode="extended",
            exportselection=False,
            height=4,
        )
        file_scroll = self.ttk.Scrollbar(
            file_list_host,
            orient="vertical",
            command=self.transient_file_list.yview,
        )
        self.transient_file_list.configure(yscrollcommand=file_scroll.set)
        self.transient_file_list.pack(side="left", fill="both", expand=True)
        file_scroll.pack(side="right", fill="y")
        file_buttons = self.ttk.Frame(files)
        file_buttons.pack(fill="x", pady=(5, 0))
        self.ttk.Button(
            file_buttons,
            text="전체 선택",
            command=self._select_all_transient_files,
        ).pack(side="left")
        self.ttk.Button(
            file_buttons,
            text="해제",
            command=self._clear_transient_files,
        ).pack(side="left", padx=(5, 0))
        self.transient_file_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_transient_run_button(),
        )

        shared = self._build_shared_component_editor(
            controls,
            "transient",
            note="Vthr 입력 시 R7/R8 재계산 · DC/Detector 탭과 같은 값",
        )
        shared.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        timing = self.ttk.LabelFrame(controls, text="Transient 설정", padding=7)
        timing.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        timing_fields = (
            ("입력 PWL Vpp [mVpp]", self.transient_input_vpp_var),
            ("output step tstep [s]", self.transient_output_step_var),
            ("stop time [s]", self.transient_stop_time_var),
            ("maximum step tmax [s]", self.transient_max_step_var),
        )
        for row, (label, variable) in enumerate(timing_fields):
            self.ttk.Label(timing, text=label).grid(
                row=row, column=0, sticky="e", padx=(0, 5), pady=2
            )
            self.ttk.Entry(timing, textvariable=variable, width=13).grid(
                row=row, column=1, sticky="ew", pady=2
            )
        timing.columnconfigure(1, weight=1)
        self.ttk.Checkbutton(
            timing,
            text="transient.csv 생성 (tran.raw와 중복, 대용량)",
            variable=self.transient_write_csv_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.ttk.Label(
            timing,
            text=(
                "입력 Vpp 기본값: 10 mVpp\n"
                "10u = 10 µs, 5u = 5 µs\n"
                "stop time 빈칸: PWL 마지막 0 V 시각\n"
                "CSV를 꺼도 그래프와 커서는 그대로 동작합니다."
            ),
            justify="left",
            foreground="#164a7b",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.ttk.Button(
            timing,
            text="step 기본값 근거",
            command=self._show_transient_setting_basis,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        channels = self.ttk.LabelFrame(controls, text="Transient 채널", padding=7)
        channels.grid(row=0, column=3, sticky="nsew", padx=(8, 0))
        channel_host = self.ttk.Frame(channels)
        channel_host.pack(fill="both", expand=True)
        self.transient_channel_list = self.tk.Listbox(
            channel_host,
            selectmode="extended",
            exportselection=False,
            height=5,
            width=22,
        )
        transient_channel_scroll = self.ttk.Scrollbar(
            channel_host,
            orient="vertical",
            command=self.transient_channel_list.yview,
        )
        self.transient_channel_list.configure(
            yscrollcommand=transient_channel_scroll.set
        )
        self.transient_channel_list.pack(side="left", fill="both", expand=True)
        transient_channel_scroll.pack(side="right", fill="y")
        channel_buttons = self.ttk.Frame(channels)
        channel_buttons.pack(fill="x", pady=(5, 0))
        self.ttk.Button(
            channel_buttons,
            text="전체 선택",
            command=self._select_all_transient_channels,
        ).pack(side="left")
        self.ttk.Button(
            channel_buttons,
            text="해제",
            command=self._clear_transient_channels,
        ).pack(side="left", padx=(5, 0))
        self.ttk.Button(
            channels,
            text="소자값 보기",
            command=lambda: self._show_channel_components(
                "transient_channel_list"
            ),
        ).pack(fill="x", pady=(4, 0))
        self.transient_channel_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._on_transient_channel_selection_changed(),
        )

        self.ttk.Label(
            outer,
            textvariable=self.transient_component_state_var,
            foreground="#9a5500",
        ).pack(fill="x", pady=(7, 0))

        action = self.ttk.Frame(outer)
        action.pack(fill="x", pady=(7, 6))
        self.transient_run_button = self.ttk.Button(
            action,
            text="Transient 실행",
            command=self._start_transient_run,
        )
        self.transient_run_button.pack(side="left")
        self.transient_stop_button = self.ttk.Button(
            action,
            text="중지",
            command=self._stop_transient_run,
            state="disabled",
        )
        self.transient_stop_button.pack(side="left", padx=(5, 0))
        self.transient_progress = self.ttk.Progressbar(
            action, mode="indeterminate", length=170
        )
        self.transient_progress.pack(side="left", padx=10)
        self.ttk.Label(action, text="결과 PWL").pack(side="left")
        self.transient_result_combo = self.ttk.Combobox(
            action,
            textvariable=self.transient_result_var,
            state="readonly",
            width=32,
        )
        self.transient_result_combo.pack(side="left", padx=(5, 8))
        self.transient_result_combo.bind(
            "<<ComboboxSelected>>",
            self._on_transient_result_changed,
        )
        self.ttk.Label(action, text="채널 ch").pack(side="left")
        result_channels = self.ttk.Frame(action)
        result_channels.pack(side="left", fill="x", expand=True)
        for channel_number in range(16):
            button = self.ttk.Radiobutton(
                result_channels,
                text=f"{channel_number:02d}",
                value=channel_number,
                variable=self.transient_result_channel_var,
                command=self._on_transient_result_channel_changed,
                state="disabled",
                width=3,
            )
            button.pack(side="left", padx=(1, 0))
            self.transient_result_channel_buttons[channel_number] = button
        self.ttk.Label(
            outer, textvariable=self.transient_status_var
        ).pack(fill="x", pady=(0, 5))

        self.transient_notebook = self.ttk.Notebook(outer)
        self.transient_notebook.pack(fill="both", expand=True)
        self.transient_plot_frame = self.ttk.Frame(self.transient_notebook)
        self.transient_log_frame = self.ttk.Frame(self.transient_notebook)
        self.transient_notebook.add(
            self.transient_plot_frame, text="Transient 결과"
        )
        self.transient_notebook.add(
            self.transient_log_frame, text="Transient 로그"
        )
        self.ttk.Label(
            self.transient_plot_frame,
            text=(
                "Transient 실행 후 v_in, v_filt, "
                "v_env + v_thr의 세 아날로그 그래프가 표시됩니다."
            ),
        ).pack(expand=True)
        self.transient_log_text = self.tk.Text(
            self.transient_log_frame, wrap="none", height=12
        )
        transient_log_y = self.ttk.Scrollbar(
            self.transient_log_frame,
            orient="vertical",
            command=self.transient_log_text.yview,
        )
        transient_log_x = self.ttk.Scrollbar(
            self.transient_log_frame,
            orient="horizontal",
            command=self.transient_log_text.xview,
        )
        self.transient_log_text.configure(
            yscrollcommand=transient_log_y.set,
            xscrollcommand=transient_log_x.set,
        )
        self.transient_log_text.grid(row=0, column=0, sticky="nsew")
        transient_log_y.grid(row=0, column=1, sticky="ns")
        transient_log_x.grid(row=1, column=0, sticky="ew")
        self.transient_log_frame.rowconfigure(0, weight=1)
        self.transient_log_frame.columnconfigure(0, weight=1)
        self._update_transient_run_button()

    def _browse_transient_folder(self) -> None:
        """Choose a converted PWL folder and populate its recursive file list."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="변환된 PWL 폴더 선택")
        if path:
            self.transient_folder_var.set(path)
            self._refresh_transient_pwl_files()

    def _refresh_transient_pwl_files(self) -> None:
        """Read every PWL file under the selected transient input folder."""

        from tkinter import messagebox

        try:
            root = Path(self.transient_folder_var.get()).expanduser().resolve()
            files = discover_pwl_files(root)
            if not files:
                raise FileNotFoundError(f"PWL 파일이 없습니다: {root}")
            self.transient_pwl_files = files
            self.transient_file_list.delete(0, "end")
            for path in files:
                self.transient_file_list.insert(
                    "end", path.relative_to(root).as_posix()
                )
            self.transient_file_list.selection_set(0)
            self.transient_status_var.set(
                f"PWL {len(files)}개 발견 · 실행할 파일을 선택하세요."
            )
            self._update_transient_run_button()
        except Exception as exc:
            self.transient_pwl_files = []
            self.transient_file_list.delete(0, "end")
            self.transient_status_var.set("PWL 폴더 읽기 실패")
            self._update_transient_run_button()
            messagebox.showerror("PWL 폴더 오류", str(exc))

    def _populate_transient_channels(self) -> None:
        """Mirror the current component-version channels into transient controls."""

        if not hasattr(self, "transient_channel_list"):
            return
        previous = {
            self.channels[index].ch
            for index in self.transient_channel_list.curselection()
            if 0 <= index < len(self.channels)
        }
        self.transient_channel_list.delete(0, "end")
        restored = False
        for index, channel in enumerate(self.channels):
            self.transient_channel_list.insert(
                "end", f"ch{channel.ch:02d} · fc={channel.f_c_hz:g} Hz"
            )
            if channel.ch in previous:
                self.transient_channel_list.selection_set(index)
                restored = True
        if self.channels and not restored:
            self.transient_channel_list.selection_set(0)
        self._on_transient_channel_selection_changed()

    def _on_transient_channel_selection_changed(self) -> None:
        """Load the first selected channel into the shared component panel."""

        self._update_transient_run_button()
        selected = [
            self.channels[index]
            for index in self.transient_channel_list.curselection()
            if 0 <= index < len(self.channels)
        ]
        if not selected:
            return
        if self._shared_component_channel("transient") != selected[0].ch:
            self._load_shared_component_fields("transient", selected[0].ch)

    def _select_all_transient_files(self) -> None:
        """Select every discovered PWL file and refresh the estimated job count."""

        self.transient_file_list.selection_set(0, "end")
        self._update_transient_run_button()

    def _clear_transient_files(self) -> None:
        """Clear PWL selection and refresh the transient run button."""

        self.transient_file_list.selection_clear(0, "end")
        self._update_transient_run_button()

    def _select_all_transient_channels(self) -> None:
        """Select all loaded channels for transient simulation."""

        self.transient_channel_list.selection_set(0, "end")
        self._on_transient_channel_selection_changed()

    def _clear_transient_channels(self) -> None:
        """Clear transient channel selection and refresh the run button."""

        self.transient_channel_list.selection_clear(0, "end")
        self._update_transient_run_button()

    def _update_transient_run_button(self) -> None:
        """Show job count and prevent an empty or concurrent transient run."""

        if not hasattr(self, "transient_run_button"):
            return
        file_count = len(self.transient_file_list.curselection())
        channel_count = len(self.transient_channel_list.curselection())
        jobs = file_count * channel_count
        running = bool(
            self.transient_worker and self.transient_worker.is_alive()
        ) or bool(self.worker and self.worker.is_alive()) or bool(
            getattr(self, "detector_worker", None)
            and self.detector_worker.is_alive()
        )
        self.transient_run_button.configure(
            text=(
                f"Transient 실행 ({jobs} jobs)"
                if jobs
                else "PWL/채널 선택 필요"
            ),
            state="disabled" if running or jobs == 0 else "normal",
        )

    def _show_transient_setting_basis(self) -> None:
        """Explain the circuit-specific transient time-step defaults."""

        from tkinter import messagebox

        highest_channel_hz = max(
            (channel.f_c_hz for channel in self.channels),
            default=6761.0,
        )
        channel_limit_us = (
            recommended_maximum_step_s(highest_channel_hz) * 1e6
        )
        speech_limit_us = recommended_maximum_step_s(8000.0) * 1e6
        messagebox.showinfo(
            "Transient 기본값 근거",
            (
                "output step tstep = 10 µs\n"
                "저장/표시용 제안 간격입니다. 1초당 명목 100,000간격이며 "
                "원래의 1e-5 s와 같은 값입니다.\n\n"
                "maximum step tmax = 5 µs\n"
                "일반적인 20 points/cycle 기준은 "
                f"최고 채널 {highest_channel_hz:.0f} Hz에서 "
                f"{channel_limit_us:.2f} µs 이하, 16 kHz WAV의 Nyquist "
                f"8 kHz에서 {speech_limit_us:.2f} µs 이하입니다. "
                "5 µs는 두 기준보다 작게 잡은 보수적인 기본값입니다.\n\n"
                "입력 PWL Vpp = 10 mVpp\n"
                "기존 PWL 파일을 수정하지 않고 실행용 사본의 전압값만 "
                "target/current 비율로 스케일합니다."
            ),
        )

    def _transient_settings(self) -> TransientSettings:
        """Parse GUI seconds/volts into a validated transient settings object."""

        stop_text = self.transient_stop_time_var.get().strip()
        try:
            input_vpp_mv = float(self.transient_input_vpp_var.get())
        except ValueError as exc:
            raise ValueError(
                "입력 PWL Vpp에는 mVpp 단위 숫자를 입력하세요."
            ) from exc
        if not math.isfinite(input_vpp_mv) or input_vpp_mv <= 0.0:
            raise ValueError("입력 PWL Vpp는 0보다 큰 유한한 값이어야 합니다.")
        settings = TransientSettings(
            output_step_s=parse_time_seconds(
                self.transient_output_step_var.get(),
                "Transient output step",
            ),
            stop_time_s=(
                parse_time_seconds(stop_text, "Transient stop time")
                if stop_text
                else None
            ),
            maximum_step_s=parse_time_seconds(
                self.transient_max_step_var.get(),
                "Transient maximum step",
            ),
            input_vpp_v=input_vpp_mv / 1000.0,
            pspice_compat=self.pspice_compat_var.get(),
            write_csv=bool(self.transient_write_csv_var.get()),
        )
        settings.validate()
        return settings

    def _start_transient_run(self) -> None:
        """Commit component edits and launch selected transient jobs."""

        from tkinter import messagebox

        if self.transient_worker and self.transient_worker.is_alive():
            return
        try:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("AC/OP 실행이 끝난 뒤 transient를 시작하세요.")
            if (
                getattr(self, "detector_worker", None)
                and self.detector_worker.is_alive()
            ):
                raise RuntimeError(
                    "Active Detector 실행이 끝난 뒤 transient를 시작하세요."
                )
            self._commit_component_editor()
            file_indices = list(self.transient_file_list.curselection())
            channel_indices = list(self.transient_channel_list.curselection())
            if not file_indices or not channel_indices:
                raise ValueError("PWL 파일과 채널을 하나 이상 선택하세요.")
            stimuli = [self.transient_pwl_files[index] for index in file_indices]
            channels = [self.channels[index] for index in channel_indices]
            settings = self._transient_settings()
            ngspice = find_ngspice(self.ngspice_var.get())
            netlist = Path(self.netlist_var.get()).expanduser().resolve()
            output = Path(self.output_var.get()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            if not netlist.is_file():
                raise FileNotFoundError(f"네트리스트를 찾을 수 없습니다: {netlist}")
        except Exception as exc:
            messagebox.showerror("Transient 실행 설정 오류", str(exc))
            return
        self.transient_log_text.delete("1.0", "end")
        # A new run supersedes the previous cache, so results are republished
        # from scratch as each job finishes instead of only at the end.
        self.transient_result_lookup = {}
        self.transient_result_stimulus_paths = {}
        self.transient_displayed_result = None
        self.transient_result_var.set("")
        self.transient_result_channel_var.set(-1)
        for button in self.transient_result_channel_buttons.values():
            button.configure(state="disabled")
        self.transient_live_rendered = False
        self.transient_completed_jobs = 0
        self.transient_total_jobs = len(stimuli) * len(channels)
        self.transient_simulator = TransientSimulator(
            lambda text: self.events.put(("tran_log", text))
        )
        self.transient_run_button.configure(state="disabled")
        self.transient_stop_button.configure(state="normal")
        self.op_run_button.configure(state="disabled")
        self.ac_run_button.configure(state="disabled")
        self.detector_run_button.configure(state="disabled")
        self.transient_progress.start(12)
        jobs = len(stimuli) * len(channels)
        self.transient_status_var.set(f"Transient {jobs} jobs 실행 중...")

        def worker() -> None:
            """Run blocking transient jobs outside Tk's event loop."""

            try:
                results = self.transient_simulator.run(
                    channels,
                    stimuli,
                    netlist,
                    output,
                    ngspice,
                    settings,
                    on_result=lambda result: self.events.put(
                        ("tran_result", result)
                    ),
                )
                self.events.put(("tran_done", results))
            except Exception as exc:
                self.events.put(("tran_error", exc))

        self.transient_worker = threading.Thread(target=worker, daemon=True)
        self.transient_worker.start()

    def _stop_transient_run(self) -> None:
        """Forward a stop request to the active transient runner."""

        if self.transient_simulator:
            self.transient_simulator.stop()
            self.transient_status_var.set("Transient 중지 요청 중...")

    def _finish_transient_controls(self) -> None:
        """Restore transient controls after success, stop, or failure."""

        self.transient_progress.stop()
        self.transient_worker = None
        self.transient_stop_button.configure(state="disabled")
        self._update_run_button()
        self._update_transient_run_button()
        self._update_detector_run_button()

    def _on_transient_result_changed(self, _event: object) -> None:
        """Enable completed channels for the selected PWL and show one result."""

        self._refresh_transient_result_channel_buttons()

    def _on_transient_result_channel_changed(self) -> None:
        """Render the completed channel clicked in the 16-channel result row."""

        key = (
            self.transient_result_var.get(),
            int(self.transient_result_channel_var.get()),
        )
        result = self.transient_result_lookup.get(key)
        # Redrawing rebuilds the plot widgets and drops every placed cursor, so
        # the already-visible result is left alone.  Job completions during a
        # run therefore cannot disturb what the user is inspecting.
        if result is not None and result is not self.transient_displayed_result:
            self._render_transient_result(result)
        self._refresh_transient_component_state()

    def _refresh_transient_result_channel_buttons(
        self,
        preferred_channel: int | None = None,
    ) -> None:
        """Enable only channels completed for the selected result PWL."""

        stimulus_label = self.transient_result_var.get()
        available = sorted(
            channel_number
            for label, channel_number in self.transient_result_lookup
            if label == stimulus_label
        )
        for channel_number, button in (
            self.transient_result_channel_buttons.items()
        ):
            button.configure(
                state="normal" if channel_number in available else "disabled"
            )
        if not available:
            self.transient_result_channel_var.set(-1)
            self._refresh_transient_component_state()
            return
        current = int(self.transient_result_channel_var.get())
        if current not in available:
            current = (
                preferred_channel
                if preferred_channel in available
                else available[0]
            )
            self.transient_result_channel_var.set(current)
        self._on_transient_result_channel_changed()

    def _transient_stimulus_label(self, path: Path) -> str:
        """Return a stable PWL dropdown label, relative to its selected root."""

        candidate = path.expanduser().resolve()
        try:
            root = Path(
                self.transient_folder_var.get()
            ).expanduser().resolve()
            return candidate.relative_to(root).as_posix()
        except (OSError, ValueError):
            return f"{candidate.parent.name}/{candidate.name}"

    def _transient_result_label(self, stimulus_path: Path) -> str:
        """Return one stimulus's dropdown label, registering it when new."""

        paths = self.transient_result_stimulus_paths
        for label, path in paths.items():
            if path == stimulus_path:
                return label
        base = self._transient_stimulus_label(stimulus_path)
        label = base
        suffix = 2
        while label in paths:
            label = f"{base} [{suffix}]"
            suffix += 1
        paths[label] = stimulus_path
        return label

    def _install_transient_result(self, result: TransientResult) -> None:
        """Publish one finished job while the remaining channels still run.

        Only the first result of a run is drawn automatically.  Later results
        just enable their channel button, so an in-progress run never replaces
        the graph, cursors, or ranges the user is currently inspecting.
        """

        label = self._transient_result_label(result.stimulus_path)
        self.transient_result_lookup[(label, result.channel.ch)] = result
        self.transient_result_combo.configure(
            values=tuple(self.transient_result_stimulus_paths)
        )
        if not self.transient_result_var.get():
            self.transient_result_var.set(label)
        if self.transient_result_var.get() == label:
            button = self.transient_result_channel_buttons.get(result.channel.ch)
            if button is not None:
                button.configure(state="normal")
        if not self.transient_live_rendered:
            self.transient_live_rendered = True
            self.transient_result_var.set(label)
            self.transient_result_channel_var.set(result.channel.ch)
            self._refresh_transient_result_channel_buttons(result.channel.ch)

    def _install_transient_results(
        self,
        results: Sequence[TransientResult],
    ) -> None:
        """Populate the PWL dropdown and 16-channel enabled result options."""

        previous_label = self.transient_result_var.get()
        previous_channel = int(self.transient_result_channel_var.get())
        for result in results:
            label = self._transient_result_label(result.stimulus_path)
            self.transient_result_lookup[(label, result.channel.ch)] = result
        labels = tuple(self.transient_result_stimulus_paths)
        self.transient_result_combo.configure(values=labels)
        if not labels:
            self.transient_result_var.set("")
            self.transient_result_channel_var.set(-1)
            self._refresh_transient_result_channel_buttons()
            return
        selected_label = (
            previous_label
            if previous_label in self.transient_result_stimulus_paths
            else labels[0]
        )
        self.transient_result_var.set(selected_label)
        self._refresh_transient_result_channel_buttons(previous_channel)

    def _refresh_transient_component_state(self) -> None:
        """Explain changed components and whether the visible result is stale."""

        if not hasattr(self, "transient_component_state_var"):
            return
        default_by_channel = {
            channel.ch: channel for channel in self.default_channels
        }
        changed_channels = [
            channel.ch
            for channel in self.channels
            if default_by_channel.get(channel.ch) != channel
        ]
        if changed_channels:
            channel_text = ", ".join(
                f"ch{channel_number:02d}"
                for channel_number in changed_channels
            )
            text = (
                f"현재 소자값: 기본 CSV 대비 변경됨 ({channel_text}) · "
                "다음 Transient 실행에 자동 적용"
            )
        else:
            text = "현재 소자값: 기본 CSV와 동일"
        try:
            editor_has_changes = self._component_editor_has_changes()
        except Exception:
            editor_has_changes = False
        if editor_has_changes:
            text += (
                " · DC 편집칸에 아직 적용하지 않은 변경 있음"
                " (Transient 실행 시 자동 적용)"
            )

        result = self.transient_result_lookup.get(
            (
                self.transient_result_var.get(),
                int(self.transient_result_channel_var.get()),
            )
        )
        if result is not None:
            try:
                current = self._channel_by_number(result.channel.ch)
            except ValueError:
                current = None
            if current == result.channel:
                text += (
                    f" · 표시 중 ch{result.channel.ch:02d} 결과에 "
                    "현재 소자값 반영됨"
                )
            else:
                text += (
                    f" · 표시 중 ch{result.channel.ch:02d} 결과는 변경 전 값"
                    " (재실행 필요)"
                )
        self.transient_component_state_var.set(text)

    def _build_transient_range_controls(self, parent: object) -> None:
        """Build a compact one-line X/Y range editor plus one status line."""

        ranges = self.ttk.LabelFrame(
            parent,
            text="표시 범위",
            padding=(6, 3),
        )
        ranges.pack(anchor="w", padx=6, pady=(0, 3))
        self.ttk.Label(ranges, text="X [ms]").grid(
            row=0, column=0, sticky="e", padx=(0, 4)
        )
        x_mode = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_x_mode_var,
            values=("자동", "수동"),
            state="readonly",
            width=5,
        )
        x_mode.grid(row=0, column=1)
        x_mode.bind(
            "<<ComboboxSelected>>",
            self._on_transient_x_mode_changed,
        )
        self.ttk.Label(ranges, text="min").grid(
            row=0, column=2, sticky="e", padx=(7, 3)
        )
        self.transient_x_min_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_x_min_var, width=9
        )
        self.transient_x_min_entry.grid(row=0, column=3)
        self.ttk.Label(ranges, text="–").grid(row=0, column=4, padx=3)
        self.transient_x_max_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_x_max_var, width=9
        )
        self.transient_x_max_entry.grid(row=0, column=5)
        for entry in (
            self.transient_x_min_entry,
            self.transient_x_max_entry,
        ):
            entry.bind(
                "<Return>",
                lambda _event: self._apply_transient_x_range(),
            )
        self.ttk.Button(
            ranges,
            text="적용",
            command=self._apply_transient_x_range,
        ).grid(row=0, column=6, padx=(5, 0))
        self.ttk.Separator(
            ranges,
            orient="vertical",
        ).grid(row=0, column=7, sticky="ns", padx=9)

        self.ttk.Label(ranges, text="Y [mV]").grid(
            row=0, column=8, sticky="e", padx=(0, 4)
        )
        y_target = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_y_target_var,
            values=("v_in", "v_filt", "v_env + v_thr"),
            state="readonly",
            width=15,
        )
        y_target.grid(row=0, column=9)
        y_target.bind(
            "<<ComboboxSelected>>",
            self._on_transient_y_target_changed,
        )
        y_mode = self.ttk.Combobox(
            ranges,
            textvariable=self.transient_y_mode_var,
            values=("자동", "수동"),
            state="readonly",
            width=5,
        )
        y_mode.grid(row=0, column=10, padx=(5, 0))
        y_mode.bind(
            "<<ComboboxSelected>>",
            self._on_transient_y_mode_changed,
        )
        self.ttk.Label(ranges, text="min").grid(
            row=0, column=11, sticky="e", padx=(7, 3)
        )
        self.transient_y_min_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_y_min_var, width=9
        )
        self.transient_y_min_entry.grid(row=0, column=12)
        self.ttk.Label(ranges, text="–").grid(row=0, column=13, padx=3)
        self.transient_y_max_entry = self.ttk.Entry(
            ranges, textvariable=self.transient_y_max_var, width=9
        )
        self.transient_y_max_entry.grid(row=0, column=14)
        for entry in (
            self.transient_y_min_entry,
            self.transient_y_max_entry,
        ):
            entry.bind(
                "<Return>",
                lambda _event: self._apply_transient_y_range(),
            )
        self.ttk.Button(
            ranges,
            text="적용",
            command=self._apply_transient_y_range,
        ).grid(row=0, column=15, padx=(5, 0))
        self.ttk.Label(
            ranges,
            textvariable=self.transient_range_status_var,
            foreground="#164a7b",
        ).grid(
            row=1,
            column=0,
            columnspan=16,
            sticky="w",
            pady=(3, 0),
        )

    def _render_transient_result(self, result: TransientResult) -> None:
        """Render large ms/mV plots with compact numeric range controls."""

        from matplotlib.figure import Figure

        configure_plot_fonts()

        self._dispose_cursor_group(self.transient_plot_cursors)
        self.transient_plot_cursors.clear()
        self.transient_axes.clear()
        self.transient_vthr_markers = []
        self.transient_canvas = None
        self.transient_displayed_result = result
        self._clear_frame(self.transient_plot_frame)
        header = self.ttk.Frame(self.transient_plot_frame, padding=(8, 4))
        header.pack(fill="x")
        self.ttk.Label(
            header,
            text=(
                f"{result.stimulus_path.parent.name}/{result.stimulus_path.name} · "
                f"ch{result.channel.ch:02d} · {len(result.rows)} points"
            ),
        ).pack(side="left")
        self._build_transient_range_controls(self.transient_plot_frame)

        plot_host = self.ttk.Frame(self.transient_plot_frame)
        plot_host.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        figure = Figure(
            figsize=(14.0, 9.2), dpi=100, constrained_layout=True
        )
        axes = figure.subplots(3, 1, sharex=True)
        times_ms, vin_mv, vfilt_mv, venv_mv, vthr_mv = (
            transient_rows_to_ms_mv(result.rows)
        )
        line_vin = axes[0].plot(
            times_ms, vin_mv, color="#1565c0", label="v_in"
        )[0]
        line_vfilt = axes[1].plot(
            times_ms, vfilt_mv, color="#00897b", label="v_filt"
        )[0]
        line_venv = axes[2].plot(
            times_ms, venv_mv, color="#1565c0", label="v_env"
        )[0]
        line_vthr = axes[2].plot(
            times_ms, vthr_mv, color="#2e7d32", label="v_thr"
        )[0]
        axes[0].set_ylabel("v_in [mV]")
        axes[1].set_ylabel("v_filt [mV]")
        axes[2].set_ylabel("v_env / v_thr [mV]")
        axes[2].set_xlabel("Time [ms]")
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        axes[2].legend(loc="best")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.28)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)

        x_low = min(times_ms)
        x_high = max(times_ms)
        if x_high <= x_low:
            x_high = x_low + 1.0
        self.transient_default_x_limits_ms = (x_low, x_high)
        # v_thr is a DC level that would otherwise dominate the shared range and
        # flatten the v_env envelope, so the automatic range follows v_env only.
        self.transient_default_y_limits_mv = {
            "v_in": data_occupancy_limits(vin_mv),
            "v_filt": data_occupancy_limits(vfilt_mv),
            "v_env + v_thr": data_occupancy_limits(venv_mv),
        }
        self.transient_axes = {
            "v_in": axes[0],
            "v_filt": axes[1],
            "v_env + v_thr": axes[2],
        }
        use_manual_x = (
            self.transient_x_mode_var.get() == "수동"
            and self.transient_manual_x_limits_ms is not None
        )
        axes[0].set_xlim(
            *(
                self.transient_manual_x_limits_ms
                if use_manual_x
                else self.transient_default_x_limits_ms
            )
        )
        for name, axis in self.transient_axes.items():
            use_manual_y = (
                self.transient_y_modes.get(name) == "수동"
                and name in self.transient_manual_y_limits_mv
            )
            axis.set_ylim(
                *(
                    self.transient_manual_y_limits_mv[name]
                    if use_manual_y
                    else self.transient_default_y_limits_mv[name]
                )
            )
        self.transient_vthr_series_mv = list(vthr_mv)
        self._annotate_offscreen_vthr(axes[2], vthr_mv)
        if self.transient_y_target_var.get() not in self.transient_axes:
            self.transient_y_target_var.set("v_in")
        self.transient_y_mode_var.set(
            self.transient_y_modes.get(
                self.transient_y_target_var.get(),
                "자동",
            )
        )

        self.transient_canvas = self._embed_multi_axes_figure(
            plot_host,
            figure,
            (
                (axes[0], (line_vin,)),
                (axes[1], (line_vfilt,)),
                (axes[2], (line_venv, line_vthr)),
            ),
        )
        for axis in axes:
            axis.callbacks.connect(
                "xlim_changed",
                self._schedule_transient_range_sync,
            )
            axis.callbacks.connect(
                "ylim_changed",
                self._schedule_transient_range_sync,
            )
        self._sync_transient_range_fields()
        self._update_transient_range_entry_states()

    def _annotate_offscreen_vthr(
        self,
        axis: object,
        vthr_mv: Sequence[float],
    ) -> None:
        """Pin v_thr to the nearest axis edge with its value when it is clipped.

        The automatic range follows v_env, so the comparator threshold often
        falls outside the view.  Rather than silently vanishing, it is drawn on
        the edge it left, labelled with its actual millivolt level.
        """

        for artist in self.transient_vthr_markers:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.transient_vthr_markers.clear()
        finite = [value for value in vthr_mv if math.isfinite(value)]
        if not finite:
            return
        low, high = axis.get_ylim()
        minimum, maximum = min(finite), max(finite)
        if minimum >= low and maximum <= high:
            return
        above = maximum > high
        level = maximum if above else minimum
        edge = high if above else low
        span = high - low
        offset = span * 0.02
        guide = axis.axhline(
            edge - offset if above else edge + offset,
            color="#2e7d32",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
        )
        label = axis.annotate(
            f"v_thr = {level:.2f} mV {'↑' if above else '↓'} (범위 밖)",
            xy=(0.99, 0.97 if above else 0.03),
            xycoords="axes fraction",
            ha="right",
            va="top" if above else "bottom",
            fontsize=9,
            color="#2e7d32",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "#e8f5e9",
                "ec": "#2e7d32",
                "alpha": 0.95,
            },
        )
        self.transient_vthr_markers.extend((guide, label))

    @staticmethod
    def _validated_axis_pair(
        minimum_text: str,
        maximum_text: str,
        label: str,
    ) -> tuple[float, float]:
        """Parse a finite, increasing min/max pair from transient controls."""

        try:
            minimum = float(minimum_text)
            maximum = float(maximum_text)
        except ValueError as exc:
            raise ValueError(f"{label}: 최소/최대값에 숫자를 입력하세요.") from exc
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"{label}: 유한한 값을 입력하세요.")
        if minimum >= maximum:
            raise ValueError(f"{label}: 최소값은 최대값보다 작아야 합니다.")
        return minimum, maximum

    def _apply_transient_x_range(self) -> None:
        """Apply only the shared X-axis auto/manual mode and limits."""

        from tkinter import messagebox

        if (
            not self.transient_axes
            or self.transient_canvas is None
            or self.transient_default_x_limits_ms is None
        ):
            return
        mode = self.transient_x_mode_var.get()
        try:
            if mode == "자동":
                self.transient_manual_x_limits_ms = None
                x_limits = self.transient_default_x_limits_ms
            elif mode == "수동":
                x_limits = self._validated_axis_pair(
                    self.transient_x_min_var.get(),
                    self.transient_x_max_var.get(),
                    "X 범위 [ms]",
                )
                self.transient_manual_x_limits_ms = x_limits
            else:
                raise ValueError("X축 모드는 자동 또는 수동이어야 합니다.")
        except ValueError as exc:
            messagebox.showerror("Transient 표시 범위 오류", str(exc))
            return
        next(iter(self.transient_axes.values())).set_xlim(*x_limits)
        self._sync_transient_range_fields(update_x=True)
        self._update_transient_range_entry_states()
        self.transient_canvas.draw_idle()
        self.transient_range_status_var.set(
            f"X축 {mode}: {x_limits[0]:.7g}–{x_limits[1]:.7g} ms"
        )

    def _apply_transient_y_range(self) -> None:
        """Apply only the selected graph's Y-axis auto/manual mode and limits."""

        from tkinter import messagebox

        if not self.transient_axes or self.transient_canvas is None:
            return
        target = self.transient_y_target_var.get()
        mode = self.transient_y_mode_var.get()
        try:
            axis = self.transient_axes[target]
            if mode == "자동":
                self.transient_manual_y_limits_mv.pop(target, None)
                y_limits = self.transient_default_y_limits_mv[target]
            elif mode == "수동":
                y_limits = self._validated_axis_pair(
                    self.transient_y_min_var.get(),
                    self.transient_y_max_var.get(),
                    "Y 범위 [mV]",
                )
                self.transient_manual_y_limits_mv[target] = y_limits
            else:
                raise ValueError("Y축 모드는 자동 또는 수동이어야 합니다.")
        except (KeyError, ValueError) as exc:
            messagebox.showerror("Transient 표시 범위 오류", str(exc))
            return
        self.transient_y_modes[target] = mode
        axis.set_ylim(*y_limits)
        if target == "v_env + v_thr":
            self._annotate_offscreen_vthr(
                axis, self.transient_vthr_series_mv
            )
        self._sync_transient_range_fields(update_x=False)
        self._update_transient_range_entry_states()
        self.transient_canvas.draw_idle()
        self.transient_range_status_var.set(
            f"{target} Y축 {mode}: "
            f"{y_limits[0]:.7g}–{y_limits[1]:.7g} mV"
        )

    def _on_transient_x_mode_changed(self, _event: object = None) -> None:
        """Enable X entries for manual mode and apply auto mode immediately."""

        self._update_transient_range_entry_states()
        if self.transient_x_mode_var.get() == "자동":
            self._apply_transient_x_range()

    def _on_transient_y_mode_changed(self, _event: object = None) -> None:
        """Remember the selected graph's Y mode and apply auto immediately."""

        target = self.transient_y_target_var.get()
        self.transient_y_modes[target] = self.transient_y_mode_var.get()
        self._update_transient_range_entry_states()
        if self.transient_y_mode_var.get() == "자동":
            self._apply_transient_y_range()

    def _on_transient_y_target_changed(self, _event: object = None) -> None:
        """Load the selected graph's independent Y mode and current limits."""

        self.transient_y_mode_var.set(
            self.transient_y_modes.get(
                self.transient_y_target_var.get(),
                "자동",
            )
        )
        self._sync_transient_range_fields(update_x=False)
        self._update_transient_range_entry_states()

    def _update_transient_range_entry_states(self) -> None:
        """Enable numeric entries only for axes currently in manual mode."""

        x_state = (
            "normal"
            if self.transient_x_mode_var.get() == "수동"
            else "disabled"
        )
        y_state = (
            "normal"
            if self.transient_y_mode_var.get() == "수동"
            else "disabled"
        )
        for entry in (
            self.transient_x_min_entry,
            self.transient_x_max_entry,
        ):
            if entry is not None:
                entry.configure(state=x_state)
        for entry in (
            self.transient_y_min_entry,
            self.transient_y_max_entry,
        ):
            if entry is not None:
                entry.configure(state=y_state)

    def _sync_transient_range_fields(
        self,
        _event: object = None,
        *,
        update_x: bool = True,
    ) -> None:
        """Mirror current Matplotlib limits into the ms/mV entry fields."""

        if not self.transient_axes:
            return
        if update_x:
            x_min, x_max = next(iter(self.transient_axes.values())).get_xlim()
            self.transient_x_min_var.set(f"{x_min:.9g}")
            self.transient_x_max_var.set(f"{x_max:.9g}")
        target = self.transient_y_target_var.get()
        axis = self.transient_axes.get(target)
        if axis is not None:
            y_min, y_max = axis.get_ylim()
            self.transient_y_min_var.set(f"{y_min:.9g}")
            self.transient_y_max_var.set(f"{y_max:.9g}")

    def _schedule_transient_range_sync(self, _axis: object) -> None:
        """Coalesce repeated shared-axis callbacks into one idle field update."""

        if self.transient_range_sync_pending:
            return
        self.transient_range_sync_pending = True

        def sync_once() -> None:
            """Clear the pending flag and mirror the latest displayed limits."""

            self.transient_range_sync_pending = False
            self._sync_transient_range_fields()

        self.root.after_idle(sync_once)

    def _embed_multi_axes_figure(
        self,
        parent: object,
        figure: object,
        axes_lines: Sequence[tuple[object, Sequence[object]]],
    ) -> object:
        """Embed one figure with button-armed line-following cursors per axes."""

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as exc:
            raise RuntimeError(
                "그래프를 표시하려면 matplotlib가 필요합니다."
            ) from exc
        canvas = FigureCanvasTkAgg(figure, master=parent)
        new_cursors: list[InteractivePlotCursor] = []
        for axes, lines in axes_lines:
            cursor = InteractivePlotCursor(
                canvas,
                axes,
                lines,
                x_name="t",
                x_unit="ms",
                y_name="V",
                y_unit="mV",
            )
            new_cursors.append(cursor)
            self.transient_plot_cursors.append(cursor)
        cursor_controls = self.ttk.Frame(parent)
        cursor_controls.pack(side="bottom", fill="x", pady=(2, 0))
        self.ttk.Button(
            cursor_controls,
            text="커서 추가",
            command=lambda: self._arm_plot_cursors(new_cursors),
        ).pack(side="left")
        self.ttk.Button(
            cursor_controls,
            text="커서 전체 삭제",
            command=lambda: self._clear_plot_cursors(new_cursors),
        ).pack(side="left", padx=(5, 0))
        self.ttk.Label(
            cursor_controls,
            text=(
                "커서 추가 → 선을 따라 이동 → 세로 보조선 위치를 클릭해 고정 "
                "· 반복 추가 가능"
            ),
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        return canvas
