"""WAV discovery and deterministic WAV-to-PWL conversion tab."""

from __future__ import annotations

import threading
from pathlib import Path

from wav_pwl import convert_wav_tree, discover_wav_files

from .base import AppController


class PwlTabController(AppController):
    """WAV discovery and deterministic WAV-to-PWL conversion tab."""

    def _build_converter_tab(self) -> None:
        """Build the WAV-tree scanner and 0 V-DC/10 mVpp conversion page."""

        outer = self.ttk.Frame(self.converter_page, padding=12)
        outer.pack(fill="both", expand=True)
        settings = self.ttk.LabelFrame(
            outer, text="WAV → ngspice PWL 변환", padding=10
        )
        settings.pack(fill="x")
        self._path_row(
            settings,
            0,
            "WAV 루트",
            self.pwl_input_var,
            self._browse_pwl_input,
        )
        self._path_row(
            settings,
            1,
            "PWL 출력",
            self.pwl_output_var,
            self._browse_pwl_output,
        )
        settings.columnconfigure(1, weight=1)
        options = self.ttk.Frame(settings)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(7, 2))
        self.ttk.Checkbutton(
            options,
            text="기존 PWL 덮어쓰기",
            variable=self.pwl_overwrite_var,
        ).pack(side="left")
        self.ttk.Button(
            options,
            text="WAV 폴더 스캔",
            command=self._scan_wav_tree,
        ).pack(side="left", padx=(10, 0))
        self.pwl_convert_button = self.ttk.Button(
            options,
            text="변환 시작",
            command=self._start_pwl_conversion,
        )
        self.pwl_convert_button.pack(side="left", padx=(5, 0))
        self.pwl_stop_button = self.ttk.Button(
            options,
            text="중지",
            command=self._stop_pwl_conversion,
            state="disabled",
        )
        self.pwl_stop_button.pack(side="left", padx=(5, 0))
        self.pwl_progress = self.ttk.Progressbar(
            options, mode="determinate", length=220
        )
        self.pwl_progress.pack(side="left", padx=(12, 0))
        self.ttk.Label(
            settings,
            text=(
                "파일별 처리: y=(x−mean)·0.010/(max−min) [V] · "
                "DC=0 V · peak-to-peak=10 mV. 마지막 원 샘플 다음 시각에 "
                "0 V를 추가해 더 긴 .tran에서도 입력을 무음으로 유지합니다."
            ),
            justify="left",
            foreground="#164a7b",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(7, 2))

        body = self.ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 0))
        words_host = self.ttk.LabelFrame(
            body, text="발견된 단어 폴더", padding=6
        )
        log_host = self.ttk.LabelFrame(body, text="변환 로그", padding=6)
        body.add(words_host, weight=1)
        body.add(log_host, weight=3)
        self.pwl_scan_table = self.ttk.Treeview(
            words_host,
            columns=("word", "count"),
            show="headings",
            height=18,
        )
        self.pwl_scan_table.heading("word", text="단어")
        self.pwl_scan_table.heading("count", text="WAV 수")
        self.pwl_scan_table.column("word", width=180, stretch=True)
        self.pwl_scan_table.column("count", width=80, anchor="e")
        scan_scroll = self.ttk.Scrollbar(
            words_host,
            orient="vertical",
            command=self.pwl_scan_table.yview,
        )
        self.pwl_scan_table.configure(yscrollcommand=scan_scroll.set)
        self.pwl_scan_table.grid(row=0, column=0, sticky="nsew")
        scan_scroll.grid(row=0, column=1, sticky="ns")
        words_host.rowconfigure(0, weight=1)
        words_host.columnconfigure(0, weight=1)
        self.pwl_log_text = self.tk.Text(log_host, wrap="none", height=18)
        pwl_log_y = self.ttk.Scrollbar(
            log_host, orient="vertical", command=self.pwl_log_text.yview
        )
        pwl_log_x = self.ttk.Scrollbar(
            log_host, orient="horizontal", command=self.pwl_log_text.xview
        )
        self.pwl_log_text.configure(
            yscrollcommand=pwl_log_y.set,
            xscrollcommand=pwl_log_x.set,
        )
        self.pwl_log_text.grid(row=0, column=0, sticky="nsew")
        pwl_log_y.grid(row=0, column=1, sticky="ns")
        pwl_log_x.grid(row=1, column=0, sticky="ew")
        log_host.rowconfigure(0, weight=1)
        log_host.columnconfigure(0, weight=1)
        self.ttk.Label(
            outer, textvariable=self.pwl_status_var
        ).pack(fill="x", pady=(7, 0))

    def _browse_pwl_input(self) -> None:
        """Choose the Speech Commands subset root and derive its PWL folder."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="WAV 데이터셋 루트 선택")
        if path:
            self.pwl_input_var.set(path)
            self.pwl_output_var.set(str(Path(path) / "pwl"))
            self._scan_wav_tree()

    def _browse_pwl_output(self) -> None:
        """Choose the parent that will contain mirrored word/PWL folders."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="PWL 출력 폴더 선택")
        if path:
            self.pwl_output_var.set(path)

    def _scan_wav_tree(self) -> None:
        """List per-word WAV counts without converting any file."""

        from tkinter import messagebox

        try:
            source_root = Path(self.pwl_input_var.get()).expanduser().resolve()
            target_root = Path(self.pwl_output_var.get()).expanduser().resolve()
            files = discover_wav_files(source_root, target_root)
            counts: dict[str, int] = {}
            for source in files:
                relative = source.relative_to(source_root)
                word = relative.parts[0] if len(relative.parts) > 1 else source_root.name
                counts[word] = counts.get(word, 0) + 1
            for item in self.pwl_scan_table.get_children():
                self.pwl_scan_table.delete(item)
            for word, count in sorted(
                counts.items(), key=lambda item: item[0].casefold()
            ):
                self.pwl_scan_table.insert(
                    "", "end", values=(word, count)
                )
            self.pwl_progress.configure(maximum=max(1, len(files)), value=0)
            self.pwl_status_var.set(
                f"{len(counts)}개 단어 폴더 · WAV {len(files)}개 발견"
            )
        except Exception as exc:
            self.pwl_status_var.set("WAV 스캔 실패")
            messagebox.showerror("WAV 폴더 스캔 오류", str(exc))

    def _start_pwl_conversion(self) -> None:
        """Convert the scanned WAV tree on a worker thread."""

        from tkinter import messagebox

        if self.conversion_worker and self.conversion_worker.is_alive():
            return
        try:
            source_root = Path(self.pwl_input_var.get()).expanduser().resolve()
            target_root = Path(self.pwl_output_var.get()).expanduser().resolve()
            wav_files = discover_wav_files(source_root, target_root)
            if not wav_files:
                raise FileNotFoundError(f"WAV 파일이 없습니다: {source_root}")
            target_root.mkdir(parents=True, exist_ok=True)
            overwrite = bool(self.pwl_overwrite_var.get())
        except Exception as exc:
            messagebox.showerror("PWL 변환 설정 오류", str(exc))
            return
        self.conversion_stop_event.clear()
        self.pwl_log_text.delete("1.0", "end")
        self.pwl_progress.configure(maximum=len(wav_files), value=0)
        self.pwl_convert_button.configure(state="disabled")
        self.pwl_stop_button.configure(state="normal")
        self.pwl_status_var.set(f"WAV {len(wav_files)}개 변환 중...")

        def progress(
            position: int,
            total: int,
            source: Path,
            target: Path,
        ) -> None:
            """Report one file before conversion and honor a stop request."""

            if self.conversion_stop_event.is_set():
                raise InterruptedError("사용자가 PWL 변환을 중지했습니다.")
            self.events.put(
                (
                    "pwl_progress",
                    (position, total, source, target),
                )
            )

        def worker() -> None:
            """Run WAV decoding and disk writes outside Tk's event loop."""

            try:
                results = convert_wav_tree(
                    source_root,
                    target_root,
                    target_vpp=0.010,
                    overwrite=overwrite,
                    progress=progress,
                )
                self.events.put(("pwl_done", results))
            except InterruptedError as exc:
                self.events.put(("pwl_stopped", exc))
            except Exception as exc:
                self.events.put(("pwl_error", exc))

        self.conversion_worker = threading.Thread(target=worker, daemon=True)
        self.conversion_worker.start()

    def _stop_pwl_conversion(self) -> None:
        """Stop before the next WAV file is converted."""

        self.conversion_stop_event.set()
        self.pwl_status_var.set("PWL 변환 중지 요청 중...")

    def _finish_pwl_controls(self) -> None:
        """Restore converter buttons after success, stop, or error."""

        self.conversion_worker = None
        self.pwl_convert_button.configure(state="normal")
        self.pwl_stop_button.configure(state="disabled")
