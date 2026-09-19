"""아날로그 전단의 소프트웨어 대응물.

파형 -> 대역 분해 -> 포락선 -> 10 ms 격자 -> 채널 문턱 -> [B, C, T] 이진 이미지.

이 경로는 학습 대상이 아니다(문턱 theta_c 만 학습한다). 실제 회로가 낼 값과
학습 입력을 맞추기 위한 모사이며, 동료의 SPICE 결과가 확정되면
filterbank="spice" 로 바꿔 같은 필터를 쓴다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .config import FrontendConfig
from .ste import step_ste

_EPS = 1e-10


class Frontend(nn.Module):
    def __init__(self, cfg: FrontendConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        self.clip_samples = int(round(sr * cfg.clip_ms / 1000.0))
        win = int(round(sr * cfg.stft_win_ms / 1000.0))
        hop = int(round(sr * cfg.stft_hop_ms / 1000.0))

        if cfg.filterbank == "mel":
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=cfg.n_fft, win_length=win, hop_length=hop,
                f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_channels, power=2.0)
        elif cfg.filterbank == "spice":
            self._spectro = torchaudio.transforms.Spectrogram(
                n_fft=cfg.n_fft, win_length=win, hop_length=hop, power=2.0)
            p = Path(cfg.spice_matrix_path).expanduser()
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            if not p.is_file():
                raise FileNotFoundError(f"spice_matrix_path={p} 가 없다.")
            m = np.loadtxt(p, delimiter=",")
            want = (cfg.n_channels, cfg.n_fft // 2 + 1)
            if m.shape != want:
                raise ValueError(
                    f"SPICE 행렬 {p} 는 {m.shape}, 기대는 {want}. "
                    "같은 STFT 격자에서 다시 뽑아야 한다.")
            self.register_buffer("spice_fbank", torch.tensor(m, dtype=torch.float32))
        else:
            raise ValueError(f"알 수 없는 filterbank {cfg.filterbank!r}")

        # 채널 문턱. 학습 대상. init_thresholds() 로 데이터 평균에 맞춰 두고 시작한다.
        self.threshold = nn.Parameter(torch.full((cfg.n_channels,), 0.5))
        # 데이터셋 전역 상수. 클립마다 바뀌지 않는다 = 절대 문턱.
        self.register_buffer("fixed_lo", torch.zeros(cfg.n_channels, 1))
        self.register_buffer("fixed_hi", torch.ones(cfg.n_channels, 1))
        # 버퍼로 둔다. state_dict() 를 재정의해 키를 끼워 넣으면 부모가 넘겨준
        # 공유 dict 에 접두사 없이 박혀 "Unexpected key(s): _scale_ready" 로
        # 체크포인트 로딩이 깨진다 (실제로 30에폭 뒤에 터졌다).
        self.register_buffer("scale_ready", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------------ 대역
    def _bands(self, wave: torch.Tensor) -> torch.Tensor:
        if self.cfg.filterbank == "mel":
            return self.melspec(wave)
        spec = self._spectro(wave)                       # [B, F, frames]
        return torch.einsum("cf,bft->bct", self.spice_fbank, spec)

    def _fix_length(self, wave: torch.Tensor) -> torch.Tensor:
        n = self.clip_samples
        if wave.shape[-1] == n:
            return wave
        if wave.shape[-1] > n:
            return wave[..., :n]
        return F.pad(wave, (0, n - wave.shape[-1]))

    # -------------------------------------------------------------- 포락선
    def envelopes(self, wave: torch.Tensor, raw: bool = False) -> torch.Tensor:
        """[B, C, native_T]. raw=True 면 정규화 전(스케일 측정용)."""
        wave = self._fix_length(wave)
        band = self._bands(wave)
        if self.cfg.compression == "log":
            band = torch.log(band + _EPS)
        elif self.cfg.compression == "sqrt":
            band = torch.sqrt(band + _EPS)               # 진폭 = 회로의 v_env
        else:
            raise ValueError(f"알 수 없는 compression {self.cfg.compression!r}")

        # 10 ms 창의 max = "그 창 안에서 한 번이라도 1이었으면 1"
        if self.cfg.envelope_reduce == "max":
            env = F.adaptive_max_pool1d(band, self.cfg.native_T)
        elif self.cfg.envelope_reduce == "mean":
            env = F.adaptive_avg_pool1d(band, self.cfg.native_T)
        else:
            raise ValueError(f"알 수 없는 envelope_reduce {self.cfg.envelope_reduce!r}")

        if raw:
            return env
        if not bool(self.scale_ready):
            raise RuntimeError(
                "init_fixed_scale() 을 먼저 부르라. lo/hi 가 데이터셋 상수여야 "
                "'env >= theta' 가 절대 문턱이 된다.")
        return (env - self.fixed_lo) / (self.fixed_hi - self.fixed_lo + _EPS)

    # -------------------------------------------------------------- 이진화
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """[B, C, T] in {0,1}. 비교기 출력에 해당한다."""
        env = self.envelopes(wave)
        theta = self.threshold.view(1, -1, 1)
        return step_ste(env - theta, self.cfg.ste_clip)

    # ---------------------------------------------------------------- 초기화
    @torch.no_grad()
    def init_fixed_scale(self, waves: torch.Tensor) -> None:
        """데이터셋 전역 lo/hi 를 채널별로 잡는다.

        hi 를 최대가 아니라 분위수로 두는 이유: 한 클립의 이상치가 전체 스케일을
        끌어올리면 나머지 클립이 전부 문턱 아래로 눌린다.
        """
        env = self.envelopes(waves, raw=True)            # [N, C, T]
        q = self.cfg.fixed_scale_quantile
        flat = env.transpose(0, 1).reshape(env.shape[1], -1)   # [C, N*T]
        self.fixed_lo.copy_(flat.amin(dim=1, keepdim=True))
        self.fixed_hi.copy_(torch.quantile(flat, q, dim=1, keepdim=True))
        self.scale_ready.fill_(True)

    @torch.no_grad()
    def init_thresholds(self, waves: torch.Tensor) -> None:
        """채널별 정규화 포락선의 평균으로 초기화 (Cerutti IV-A)."""
        env = self.envelopes(waves)                      # [N, C, T]
        self.threshold.copy_(env.mean(dim=(0, 2)))
