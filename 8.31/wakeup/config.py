"""설정. 하드웨어로 넘어가는 값과 학습만의 값을 한 곳에 모아 둔다.

회로가 거는 제약(10 ms 격자, 상태 수, timeout)은 전부 여기서 강제한다.
실장할 수 없는 해가 학습에서 나오지 않게 하려면 제약이 모델보다 앞에 있어야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class FrontendConfig:
    """아날로그 전단의 소프트웨어 대응물. 학습 대상은 채널 문턱뿐이다."""

    sample_rate: int = 16000
    clip_ms: float = 1000.0

    # STFT — 아날로그 필터뱅크를 흉내내는 공통 프론트엔드
    stft_win_ms: float = 25.0
    stft_hop_ms: float = 10.0
    n_fft: int = 512

    # 대역 분해
    n_channels: int = 8            # 실장 채널 수. 8/16 미확정이라 config 로 둔다
    f_min: float = 125.0           # 동료 보드 실측 대역
    f_max: float = 5000.0
    # "mel" = 소프트웨어 근사, "spice" = 동료 SPICE 행렬 CSV
    filterbank: str = "mel"
    spice_matrix_path: str = ""    # [C, n_fft//2+1] CSV

    # 포락선
    # log 는 단조라서 "정규화 로그값 >= theta" 와 "전압 >= V_th" 가 동치다.
    # 즉 고정 문턱(R7/R8 분압)을 그대로 표현한다.
    compression: str = "log"       # log | sqrt
    envelope_win_ms: float = 10.0  # = 회로의 100 Hz 격자
    envelope_reduce: str = "max"   # 2.8 의 "한 번이라도 1이면 1"

    # 정규화 — fixed 만 허용한다. 클립마다 스케일이 변하면 저항비로 못 만든다.
    normalize: str = "fixed"
    fixed_scale_quantile: float = 0.75

    # 문턱 초기화의 목표 켜짐률. 0 이면 채널 평균(Cerutti IV-A)으로 둔다.
    # 실제 음성에서 평균은 곧 분포 한가운데라 켜짐률이 57% 가 되고, 그러면
    # 이진 이미지가 거의 무작위가 되어 어느 시각의 AUC 도 0.56 을 못 넘는다.
    # 회로에서는 VTHR 을 골라 채널 듀티를 정하는 것에 해당한다.
    init_on_rate: float = 0.0

    # 문턱 이진화. 이 값이 문턱이 실제로 학습되는지를 좌우한다 -- |env-theta| 가
    # clip 안에 있을 때만 기울기가 흐르므로, 너무 좁으면 문턱이 초기값에 묶이고
    # 너무 넓으면 특징이 빽빽해진다. 합성 과제에서 재어 본 값:
    #   0.003 -> acc 0.855 (문턱 거의 안 움직임)
    #   0.03  -> acc 0.918  <- 채택
    #   0.1   -> acc 0.680 (켜짐률 21% -> 43%)
    # 실제 음성으로 옮길 때 다시 재야 한다.
    ste_clip: float = 0.03

    @property
    def native_T(self) -> int:
        return int(round(self.clip_ms / self.envelope_win_ms))


@dataclass
class HeadConfig:
    """형판 정합부. 여기서 나오는 값이 그대로 저항과 배선이 된다."""

    n_states: int = 4              # 상태 하나 = 비교기 1 + 플립플롭 1 + 저항 한 벌
    tau: List[int] = field(default_factory=lambda: [8, 16, 24, 32])
    timeout: int = 48              # > tau[-1], 그리고 <= 63 (디코더 6비트 별칭)

    # 채점 창. tau(s) 한 순간이 아니라 tau(s) ± match_window 안에서 한 번이라도
    # 맞으면 통과로 본다.
    #
    # 왜 필요한가: START 에서 멀어질수록 정렬 오차가 쌓여 먼 시각이 못 쓰게 되고,
    # 그래서 탐색이 tau 를 START 근처로 몰아넣는다(실측: 네 시각이 110 ms 안).
    # 시각들이 몰리면 상태가 상관되어 AND 가 FPR 을 못 줄이고, 그걸 맞추려고
    # k 가 극단으로 올라가 TPR 이 무너진다.
    #
    # 회로에서는 PASS 를 에지 트리거 D 플립플롭이 아니라 **셋 우세 래치**로 두고,
    # 디코더 출력 2w+1 개를 OR 해 창을 만든다. 디코더는 모든 출력을 이미
    # 갖고 있으므로 부품이 거의 늘지 않는다.
    match_window: int = 0

    # 판정 급경사도. 작을수록 계단에 가깝다(추론과 일치), 크면 기울기가 산다.
    temperature: float = 0.5
    # AND 의 부드럽기. WAKE 는 가장 약한 상태가 정하므로 min 의 완화형을 쓴다.
    and_temperature: float = 0.5

    # 형판 게이트 g: 0 이면 X(미연결). 학습 초기에는 전부 사용으로 시작한다.
    # clip 경계(1.0)에 두면 한 번 밖으로 밀린 게이트가 영영 기울기를 못 받는다.
    gate_init: float = 0.5
    gate_ste_clip: float = 1.0
    weight_ste_clip: float = 1.0

    l1_gate: float = 0.02          # X 를 늘리는 압력 -> 저항 수 감소
    # L1 을 학습 처음부터 걸면 형판이 아무것도 배우기 전에 채널을 뜯어낸다.
    # 실측: 3에폭 만에 저항 64 -> 26, 그 결과 상태 셋이 m=3~4 로 쪼그라들고
    # k=1(무조건 통과)이 되어 s1 혼자 일했다. 이 비율만큼은 L1 을 끄고 간다.
    l1_warmup_frac: float = 0.35
    # WAKE 최소 폭. timeout 을 tau[-1] 바로 뒤에 붙이면 펄스가 한 프레임뿐이라
    # tau 가 조금만 밀려도 사라진다. timeout >= tau[-1] + 이 값.
    min_wake_frames: int = 5
    # k 는 정수 판정 문턱이라 경사하강이 아니라 탐색으로 맞춘다. 다만 학습
    # **중에** 자주 다시 맞추면 동작점이 계속 갈아엎혀 형판이 어떤 k 에도
    # 적응하지 못한다(실측: 50스텝마다 = 에폭 30개에 약 2000번, k 가
    # [16,0,0,0] -> [0,16,14,0] -> [16,16,16,16] 으로 요동). 0 이면 에폭
    # 끝에서만 맞춘다.
    refit_k_every: int = 0
    # k 탐색은 "FPR 을 이 값 이하로 묶고 그 안에서 TPR 최대화". 가중합을 쓰면
    # 전부 거부하는 해가 최적이 되어 버린다(head.fit_k 주석 참고).
    k_max_fpr: float = 0.05
    min_channels: int = 3          # 상태당 최소 사용 채널. 너무 줄면 잡음에 뜬다
    min_channels_weight: float = 1.0

    def validate(self, front: FrontendConfig) -> None:
        if len(self.tau) != self.n_states:
            raise ValueError(
                f"tau 가 {len(self.tau)}개인데 n_states 는 {self.n_states}다.")
        if sorted(self.tau) != list(self.tau):
            raise ValueError(f"tau 는 오름차순이어야 한다: {self.tau}")
        if self.tau[0] < 1:
            raise ValueError(
                "tau[0] >= 1 이어야 한다. IDLE 에서 카운터가 0 에 묶여 있어 "
                "0 은 디코드해도 뜨지 않는다.")
        if self.timeout <= self.tau[-1]:
            raise ValueError(
                f"timeout({self.timeout}) > tau[-1]({self.tau[-1]}) 이어야 한다. "
                "마지막 채점 전에 CLR 되면 WAKE 가 나올 수 없다.")
        if self.timeout > 63:
            raise ValueError(
                f"timeout({self.timeout}) > 63. 디코더를 3개 쓰지 않으면 상위 "
                "2비트가 판정에 안 들어가 64 카운트마다 같은 신호가 또 뜬다. "
                "디코더 3개 구성이면 이 검사를 풀어도 된다.")
        if self.tau[-1] >= front.native_T:
            raise ValueError(
                f"tau[-1]({self.tau[-1]}) 가 클립 길이({front.native_T} 프레임) "
                "밖이다.")


@dataclass
class StartConfig:
    """시간 원점. 학습이 아니라 탐색으로 정한다."""

    # sum_c x[c,t] >= k 인 첫 프레임을 START 로 본다
    k: int = 2
    # 그 조건이 연속 몇 프레임 이어져야 START 로 인정할지. 1 이면 순간 조건.
    # 실제 음성에서는 배경잡음만으로도 k 개가 순간적으로 켜져 프레임 0 근처에서
    # START 가 떠 버린다. 그러면 tau 가 상대 시각이 아니라 절대 시각이 되어
    # 단어 위치 변동을 전혀 보정하지 못한다.
    min_frames: int = 1
    # START 는 100 Hz 클럭과 비동기다 -> 학습 중 프레임 지터를 넣는다
    jitter: int = 1


@dataclass
class TrainConfig:
    target_word: str = "on"
    batch_size: int = 128
    epochs: int = 30
    lr: float = 3e-3
    lr_threshold: float = 1e-3     # 문턱은 스케일이 달라 따로 준다
    weight_decay: float = 0.0
    seed: int = 0
    # 양성 가중. 1.0 이 기본이다.
    #
    # 자동(N/P 비 = 약 4.2)으로 두었더니 손실은 "양성을 통과시켜라"로 밀고
    # fit_k 는 "FPR 상한을 지켜라"로 밀어 정확히 반대 방향으로 싸웠다.
    # 손실은 분리도만 만들면 되고, 동작점은 fit_k 가 고른다. 둘을 분리한다.
    # 0 이면 자동(N/P 비) -- 이제 권장하지 않는다.
    pos_weight: float = 1.0
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class Config:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    start: StartConfig = field(default_factory=StartConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_root: str = "~/datasets/speech_commands_v2"
    out_dir: str = "runs"
    tag: str = "wk_base"

    def validate(self) -> None:
        if self.frontend.normalize != "fixed":
            raise ValueError(
                "normalize 는 fixed 여야 한다. 클립마다 스케일이 바뀌면 "
                "비교기 기준전압을 저항 분압으로 고정할 수 없다.")
        self.head.validate(self.frontend)

    def to_dict(self) -> dict:
        return asdict(self)
