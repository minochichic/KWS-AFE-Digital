"""Config dataclasses for BinaryMatchboxNet.

Design notes
------------
* YAML is the single source of truth; dataclasses give type hints + validation.
  Nothing downstream (models/, data/, export/) may hardcode C, T, kernels or
  precision -- see CLAUDE.md section 5.
* Layer precision is a *config flag*, not a code branch scattered around the
  model, so the Conv2 binary-vs-int8 ablation (CLAUDE.md 2.2) is a one-line
  YAML change.
* `validate()` encodes CLAUDE.md's two hard rules (first/last layer never
  binary) so a bad YAML fails loudly at load time instead of silently training
  an architecture we promised not to build.

Python 3.9-compatible (Colab is newer, but local dev here is 3.9).
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Precision tags understood by models/. "fp32" is for debugging only.
PRECISIONS = ("binary", "int8", "fixed", "fp32")


# --------------------------------------------------------------------------- #
# AFE (analog front-end simulation)
# --------------------------------------------------------------------------- #
@dataclass
class AFEConfig:
    """Software simulation of Cerutti et al.'s analog front end.

    Pipeline: waveform -> STFT -> mel filterbank built directly with
    n_channels triangular filters (corner freqs equally spaced in the mel
    domain, CLAUDE.md 3) -> log -> per-window envelope (max) -> min-max
    normalize -> per-channel learnable threshold -> {-1,+1}.

    n_mels is NOT used by the AFE path; it is the analysis resolution for the
    full-precision Mel baseline (Cerutti VI-A) we may compare against.

    Cerutti IV-A: envelope = "maximum full-precision values of the spectrogram
    in windows of 10 ms or 25 ms"; thresholds initialized with the per-channel
    average over the training set, min-max scaled the same way as the features.
    """

    n_channels: int = 16          # AFE binary channels -> model input rows
    sample_rate: int = 16000
    clip_ms: float = 1000.0       # Speech Commands clips are 1 s

    # STFT front-end (MatchboxNet 4.1 / Cerutti VI-A both use 25 ms / 10 ms)
    stft_win_ms: float = 25.0
    stft_hop_ms: float = 10.0
    n_fft: int = 512

    # Mel filterbank. n_mels is the *analysis* resolution; n_channels is the
    # AFE filter count. These are deliberately different (CLAUDE.md 3).
    n_mels: int = 64          # NOT used by the AFE path (it builds n_channels
                              # filters); placeholder for a full-precision Mel
                              # baseline. See docs/afe_config.md.
    f_min: float = 50.0
    f_max: float = 8000.0     # <= Nyquist; Cerutti's best range is 50 Hz-8 kHz

    # Phase B (circuit-matched front end). filterbank_source selects the AFE
    # filter shapes: "mel" = ideal triangular bank (baseline); "spice" = the
    # SPICE-extracted GIC bank (AFE/artifacts/filterbank_matrix.csv, [n_channels,
    # n_fft//2+1], per-channel peak-normalized). "spice" tests the REAL filter
    # response (wider 2nd-order skirts / more channel overlap; cosine 0.83 vs
    # mel) -- see AFE/README.md "16채널 필터뱅크". global min-max is kept either
    # way. NOTE: the matrix is peak-normalized per channel, so per-channel gain
    # (6-7.8 dB spread) is equalized; a gain-weighted variant is a follow-up.
    filterbank_source: str = "mel"          # "mel" | "spice"
    spice_matrix_path: str = "AFE/artifacts/filterbank_matrix.csv"
    # The SPICE matrix is per-channel peak-normalized (gain equalized). With
    # this flag the "spice" bank re-weights each channel by its true linear
    # passband gain (from AFE/artifacts/filterbank_design.csv gain_dB), so the
    # real spectral tilt survives instead of every channel peaking at 1. NOTE:
    # the gain spread is only ~1.8 dB (x1.23 amp), and per-channel learnable
    # thresholds + global min-max absorb most of it, so expect a small effect.
    spice_gain_restore: bool = False

    # Stage-2 circuit fidelity: the analog detector has a DEADZONE (precision
    # rectifier can't rectify signals below ~a few mV, slew-limited). Modeled as
    # a learnable per-channel floor on the amplitude: relu(sqrt(power) - dz_c),
    # applied after compression, before EMA/normalize. dz_c inits to 0 (exact
    # no-op = baseline) and trains end-to-end, so the net learns how much of each
    # channel's broad-skirt tail to cut. Local probe: a deadzone drops the
    # inter-channel correlation 0.58->0.38 (sharpens the real 2nd-order filters).
    # Use with compression="sqrt" (on log it would clip negative values).
    spice_deadzone: bool = False

    # Comparator input offset Vos, as a std in the normalized-envelope domain
    # (the real LPV7215 has ~mV offset; the ideal sign() has none). Injected at
    # the comparison so it jitters each channel's effective threshold. A channel
    # whose normalized dynamic range is <~ vos gets corrupted -- the HF-margin
    # problem the mic pre-amp fixes. Set vos ~ Vos/global_V+swing: e.g. no-preamp
    # ~0.10, with-preamp ~0.035. 0.0 = exact baseline (ideal comparator).
    comparator_vos: float = 0.0


    # Envelope compression. "log" = log-mel convention (baseline). "sqrt" =
    # amplitude (V+ ~ sqrt(power)), which is what the analog active detector
    # actually does. With the broad-skirt SPICE bank, "log" lifts the filter
    # tails and makes channels ~0.9 correlated (redundant); "sqrt" is both
    # circuit-faithful AND drops inter-channel correlation below mel (~0.56 vs
    # 0.77). Recommended for filterbank_source="spice". See AFE/README.md.
    compression: str = "log"                # "log" | "sqrt"

    # AFE envelope window: 10 ms or 25 ms (Cerutti IV-A). Sets the native T.
    envelope_win_ms: float = 25.0
    envelope_reduce: str = "max"  # "max" | "mean"

    # Envelope smoothing time constant (active-detector C3 model). CLAUDE.md 3.2:
    # baseline OFF (0.0 = exact no-op). NOT a paper value -- a design DOF for a
    # future ablation. The smoothing step (EMA over STFT frames) is not yet
    # implemented in data/afe.py; this field only reserves the interface.
    envelope_tau_ms: float = 0.0

    # Binarization
    # "minmax" = per-clip, channel-shared (our original; an idealized non-causal
    #            AGC -- no fixed-divider hardware counterpart).
    # "fixed"  = dataset-level lo/hi computed ONCE (call init_fixed_scale()).
    #            This is what Cerutti IV-A describes ("the same min-max values are
    #            used to scale the initial thresholds ... adapted by selecting the
    #            corresponding resistor divider"). Being an affine map it is
    #            EQUIVALENT to an absolute threshold, so it maps directly to a
    #            FIXED R7/R8 -- while keeping envelopes ~[0,1] so ste_clip and the
    #            learning rate need no retuning. Values may exceed 1 on clips
    #            louder than the calibration set; that is meaningful (absolute
    #            scale) and is NOT clamped.
    # "none"   = no scaling (absolute). Same decisions as "fixed" but envelopes
    #            span ~[0.001, 69], so ste_clip MUST be raised (~4.0) or ~45% of
    #            elements fall outside the STE window and get no gradient.
    # "agc"    = causal, channel-shared automatic gain control: divide by a
    #            running level (fast attack / slow release) instead of the
    #            clip's future-peeking max. This is the HARDWARE-REALIZABLE
    #            version of "minmax" -- a single shared gain control, which is
    #            exactly what per-clip channel-shared min-max is an idealization
    #            of. Needs init_fixed_scale() for the reference/floor.
    # "xmax"   = cross-channel relative threshold: divide by the INSTANTANEOUS
    #            max across the 16 channels. A gain g multiplies numerator and
    #            denominator alike, so it cancels exactly -- level invariance is
    #            structural, not learned. Hardware: a 16-diode OR of the envelope
    #            outputs gives that max passively, feeding one shared divider, so
    #            unlike AGC there is NO feedback loop, NO time constant, and no
    #            stability / first-word risk (and it replaces the 16 per-channel
    #            dividers, saving their ~52 uW). Needs init_fixed_scale() for the
    #            silence floor.
    normalize: str = "minmax"           # "minmax" | "fixed" | "agc" | "xmax" | "none"

    # normalize="xmax" silence floor, as a QUANTILE of the per-frame
    # cross-channel max (measured by init_fixed_scale). In pure relative mode a
    # silent frame divides noise by noise and fires at random, so the denominator
    # is floored: frames above it stay relative (gain-invariant), quieter ones
    # fall back to an absolute threshold. The hardware gets this for free -- the
    # diode-OR node cannot fall below the detector's quiescent level.
    # Anchoring to a quantile of FRAMES matters: a fraction of the dataset peak
    # put the floor ~200x above a median frame and bound 87% of them, which
    # silently degenerates this mode into "fixed". 0.10 = only the quietest 10%
    # of frames use the absolute path; measured gain invariance (binary output
    # unchanged under an input gain) then runs 96.8-98.3% for g in [0.5, 4] vs
    # 87-95% for "fixed". Lowering it to 0.02 reaches 99.5% but leaves almost no
    # absolute floor to keep silence quiet.
    xmax_floor_frac: float = 0.10

    # AGC loop (normalize="agc" only). Fast attack tracks onsets; slow release
    # holds the level through a word. agc_max_gain_db caps the gain so silence
    # does not get amplified into noise (a real AGC's noise gate): the level
    # estimate is floored at fixed_hi / 10^(dB/20).
    # normalize="fixed"/"agc" scale reference, set by init_fixed_scale().
    # 1.0 = global max (original behaviour). <1.0 = that quantile of the
    # PER-CLIP maxima, which is outlier-robust: the global max is a single loud
    # clip that compresses typical clips into a sliver of [0,1] and leaves the
    # learned thresholds too small for the shared learning rate. Does not change
    # the achievable optimum (affine scale absorbed by the thresholds), only the
    # optimization conditioning. ~0.75 matches per-clip min-max's spread.
    fixed_scale_quantile: float = 1.0

    agc_attack_ms: float = 10.0
    agc_release_ms: float = 250.0
    agc_max_gain_db: float = 20.0
    threshold_init: str = "channel_mean"
    threshold_trainable: bool = True
    ste: str = "hardtanh"               # STE flavor for the step function
    ste_clip: float = 1.0

    @property
    def native_T(self) -> int:
        """Number of envelope windows produced by a full-length clip."""
        return int(round(self.clip_ms / self.envelope_win_ms))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class StageConfig:
    """One prologue/epilogue conv, or one residual block of R sub-blocks.

    `channels_mult` is relative to the model width C so that a single C knob
    scales the whole net (MatchboxNet Table 1 uses 2*C for Conv1/2/3).
    `channels_abs`, when set, overrides it (used by Conv4 = n_classes).
    """

    name: str
    kernel: int
    precision: str
    stride: int = 1
    dilation: int = 1
    channels_mult: float = 1.0
    channels_abs: Optional[int] = None
    n_sub_blocks: int = 1          # R, only >1 for the TCS residual blocks
    separable: bool = False        # True -> depthwise + pointwise (TCS)
    residual: bool = False
    dropout: float = 0.0

    def out_channels(self, C: int, n_classes: int) -> int:
        if self.channels_abs is not None:
            return self.channels_abs
        return max(1, int(round(C * self.channels_mult)))


@dataclass
class ModelConfig:
    """BinaryMatchboxNet-BxRxC, per CLAUDE.md 2.2."""

    C: int = 64                   # channel width -- primary sweep axis
    T: int = 64                   # time window count -- primary sweep axis
    n_classes: int = 12
    in_channels: int = 16         # must equal afe.n_channels

    # Binary weight/activation handling (QAT)
    weight_ste: str = "hardtanh"
    weight_ste_clip: float = 1.0
    scale_binary_weights: bool = True   # per-output-channel alpha (XNOR-Net)

    bn_momentum: float = 0.1
    bn_eps: float = 1e-5
    final_pool: str = "avg"

    stages: List[StageConfig] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Data / training
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    name: str = "speech_commands_v2"
    root: str = "datasets/speech_commands_v2"
    version: str = "v0.02"
    # 12-class setup (Cerutti IV-A)
    keywords: List[str] = field(default_factory=lambda: [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    ])
    unknown_label: str = "_unknown_"
    silence_label: str = "_silence_"
    silence_fraction: float = 0.1     # share of synthesized silence clips
    unknown_fraction: float = 0.1
    # Official split via validation_list.txt / testing_list.txt hashing
    split: str = "official"
    num_workers: int = 2
    cache_features: bool = True

    # Augmentation (waveform, train split only). OFF by default so the base
    # config reproduces the no-aug baseline byte-for-byte; turn on via override.
    aug_time_shift_ms: float = 0.0     # random shift in +/- this (0 = off)
    aug_noise_prob: float = 0.0        # per-sample prob of adding background noise
    aug_noise_snr_db: List[float] = field(default_factory=lambda: [5.0, 30.0])
    # Random per-clip loudness, in dB, applied to the waveform (0,0 = off).
    # NOT a MatchboxNet augmentation -- its MFCC pipeline absorbs level. Ours
    # cannot: with normalize="fixed" the comparator threshold is a fixed
    # voltage, so clip loudness leaks straight into the binary image (measured
    # as the dominant loss, ~-8pp). Training across gains is the only
    # purely-software lever against it. e.g. [-10, 10] = x0.32..x3.2.
    aug_gain_db: List[float] = field(default_factory=lambda: [0.0, 0.0])
    aug_specaug_time_masks: int = 0    # SpecAugment: deferred (not implemented)
    aug_specaug_freq_masks: int = 0


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "adam"       # "adam" | "novograd" (phase 2)
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "plateau"    # "plateau" | "warmup_hold_decay" | "none"
    warmup_ratio: float = 0.05    # WHD only (MatchboxNet 4.1)
    hold_ratio: float = 0.45
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    amp: bool = True
    seed: int = 1234
    log_every: int = 50
    out_dir: str = "runs"
    device: str = "auto"


@dataclass
class Config:
    afe: AFEConfig = field(default_factory=AFEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tag: str = "base"

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        m, a = self.model, self.afe

        if m.in_channels != a.n_channels:
            raise ValueError(
                f"model.in_channels ({m.in_channels}) must equal "
                f"afe.n_channels ({a.n_channels})"
            )
        if a.envelope_win_ms <= 0:
            raise ValueError(
                f"afe.envelope_win_ms must be positive, got {a.envelope_win_ms}"
            )
        if a.envelope_win_ms not in (10.0, 25.0):
            # 10/25 ms are the paper's values (Cerutti IV-A); anything else is a
            # deliberate deviation. Shrinking the window is a real lever -- it
            # raises the input bit count without touching the analog front end
            # (the comparator events already have sub-ms resolution) -- so warn
            # instead of blocking, and let time_axis_report() catch a T that
            # starves conv2.
            warnings.warn(
                f"afe.envelope_win_ms={a.envelope_win_ms} is not a paper value "
                f"(Cerutti IV-A uses 10 or 25 ms). Intentional deviation? "
                f"native_T = clip_ms/envelope_win_ms = "
                f"{a.clip_ms / a.envelope_win_ms:.0f}; set model.T to match.",
                stacklevel=2,
            )
        if a.f_max > a.sample_rate / 2:
            raise ValueError(
                f"afe.f_max ({a.f_max}) exceeds Nyquist ({a.sample_rate / 2})"
            )
        if not m.stages:
            raise ValueError("model.stages is empty")

        names = [s.name for s in m.stages]
        for s in m.stages:
            if s.precision not in PRECISIONS:
                raise ValueError(
                    f"stage {s.name}: unknown precision {s.precision!r}, "
                    f"expected one of {PRECISIONS}"
                )

        # --- CLAUDE.md 2.2 absolute rules -------------------------------- #
        first, last = m.stages[0], m.stages[-1]
        if first.precision == "binary":
            raise ValueError(
                f"CLAUDE.md 2.2: the first layer ({first.name}) must never be "
                f"binarized -- the input is already binary."
            )
        if last.precision == "binary":
            raise ValueError(
                f"CLAUDE.md 2.2: the last layer ({last.name}) must never be "
                f"binarized -- it needs fixed-point 12-class separability."
            )
        if last.out_channels(m.C, m.n_classes) != m.n_classes:
            raise ValueError(
                f"final stage {last.name} must emit n_classes="
                f"{m.n_classes} channels"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")

    def warnings(self) -> List[str]:
        """Non-fatal consistency notes (T vs. AFE window, kernel spans)."""
        out: List[str] = []
        native = self.afe.native_T
        if self.model.T != native:
            verb = "zero-padded" if self.model.T > native else "cropped"
            out.append(
                f"model.T={self.model.T} != native T={native} from a "
                f"{self.afe.envelope_win_ms:.0f} ms envelope window over a "
                f"{self.afe.clip_ms:.0f} ms clip -> features will be {verb}."
            )
        out.extend(self.time_axis_report()[1])
        return out

    def time_axis_report(self):
        """Track the time-axis length through the stages.

        MatchboxNet's kernels (up to k=29, dilation=2 -> span 57) were sized for
        T=128 frames. Small (C, T) sweep points shrink the time axis below the
        kernel span, at which point a layer convolves mostly over padding --
        wasted parameters and a likely accuracy cliff. Flag it rather than
        silently training it.
        """
        lengths: List[tuple] = []
        notes: List[str] = []
        t = self.model.T
        for s in self.model.stages:
            t_in = t
            t = -(-t // s.stride)  # ceil div; 'same' padding
            span = (s.kernel - 1) * s.dilation + 1
            lengths.append((s.name, t_in, t, span))
            if span > t_in:
                notes.append(
                    f"stage {s.name}: kernel span {span} "
                    f"(k={s.kernel}, dilation={s.dilation}) exceeds its input "
                    f"time length {t_in} -> convolves mostly over padding."
                )
        return lengths, notes

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _build(cls, raw: Optional[Dict[str, Any]]):
    """Instantiate a dataclass from a dict, rejecting unknown keys.

    Silently dropping a typo'd key would mean training a different model than
    the YAML claims, so unknown keys are a hard error.
    """
    if raw is None:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown config key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    return cls(**raw)


def load_config(path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config.

    `overrides` uses dotted paths, e.g. {"model.C": 32, "model.T": 40} -- this
    is what the (C, T) sweep in experiments/ drives.
    """
    with Path(path).open() as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    model_raw = dict(raw.get("model") or {})
    stages_raw = model_raw.pop("stages", None) or []

    cfg = Config(
        afe=_build(AFEConfig, raw.get("afe")),
        model=_build(ModelConfig, model_raw),
        data=_build(DataConfig, raw.get("data")),
        train=_build(TrainConfig, raw.get("train")),
        tag=raw.get("tag", "base"),
    )
    cfg.model.stages = [_build(StageConfig, s) for s in stages_raw]

    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)

    cfg.validate()
    return cfg


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise ValueError(f"override {dotted!r}: no such section {p!r}")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise ValueError(f"override {dotted!r}: no such key {leaf!r}")
    setattr(obj, leaf, value)
