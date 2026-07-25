"""Speech Commands v2 12-class pipeline tests (FIRST_TASK.md step 6).

The real 2.3 GB download happens only on Colab. Here we test the *logic* that
turns 35 raw words into the 12-class setup, with in-memory fake items -- no
torchaudio dataset download. What must hold:

1. Label mapping: the 10 keywords -> 0..9, everything else -> unknown, plus a
   silence class. Order is stable (export/hardware depend on it).
2. unknown is subsampled to ~keyword-class size, not left to swamp the set
   (Cerutti IV-A: unknown is a *subset* of the remaining 25 words).
3. silence clips are synthesized from background noise, 1 s, in {-1,+1} after
   the AFE.
4. Fixed-length collation pads/crops every waveform to exactly 1 s so a batch
   stacks and the AFE can consume it.
5. Splits stay disjoint and deterministic under a fixed seed.
"""

from __future__ import annotations

import math

import pytest
import torch

from data.speech_commands import (
    KEYWORDS,
    SILENCE_INDEX,
    UNKNOWN_INDEX,
    FixedLengthCollate,
    SpeechCommands12,
    class_names,
    label_to_index,
    subsample_indices,
    synthesize_silence,
)
from train.config import DataConfig


# --------------------------------------------------------------------------- #
# 1. label mapping
# --------------------------------------------------------------------------- #
def test_class_names_are_12_and_ordered() -> None:
    names = class_names()
    assert len(names) == 12
    assert names[:10] == KEYWORDS
    assert names[SILENCE_INDEX] == "_silence_"
    assert names[UNKNOWN_INDEX] == "_unknown_"


def test_keywords_map_to_their_position() -> None:
    for i, kw in enumerate(KEYWORDS):
        assert label_to_index(kw) == i


def test_non_keyword_maps_to_unknown() -> None:
    for word in ("bed", "bird", "house", "marvin", "wow", "three"):
        assert label_to_index(word) == UNKNOWN_INDEX


def test_silence_word_maps_to_silence() -> None:
    assert label_to_index("_silence_") == SILENCE_INDEX
    assert UNKNOWN_INDEX != SILENCE_INDEX


# --------------------------------------------------------------------------- #
# 2. unknown subsampling
# --------------------------------------------------------------------------- #
def test_subsample_is_capped_and_deterministic() -> None:
    idx = subsample_indices(n_total=1000, n_keep=50, seed=0)
    assert len(idx) == 50
    assert len(set(idx)) == 50                       # no duplicates
    assert max(idx) < 1000
    assert idx == subsample_indices(1000, 50, seed=0)         # deterministic
    assert idx != subsample_indices(1000, 50, seed=1)         # seed matters


def test_subsample_keep_ge_total_returns_all() -> None:
    idx = subsample_indices(n_total=30, n_keep=50, seed=0)
    assert sorted(idx) == list(range(30))


# --------------------------------------------------------------------------- #
# 3. silence synthesis
# --------------------------------------------------------------------------- #
def test_synthesize_silence_shape_and_length() -> None:
    noise = [torch.randn(50000), torch.randn(48000)]
    clips = synthesize_silence(noise, n=5, length=16000, seed=0)
    assert clips.shape == (5, 16000)


def test_synthesize_silence_includes_pure_zeros() -> None:
    """Warden's split reserves some fully-silent clips, not only noise."""
    noise = [torch.randn(40000)]
    clips = synthesize_silence(noise, n=20, length=16000, seed=0,
                               zero_fraction=0.25)
    n_zero = sum(1 for c in clips if torch.count_nonzero(c) == 0)
    assert n_zero > 0


def test_synthesize_silence_is_deterministic() -> None:
    noise = [torch.randn(40000)]
    a = synthesize_silence(noise, 8, 16000, seed=3)
    b = synthesize_silence(noise, 8, 16000, seed=3)
    assert torch.equal(a, b)


def test_synthesize_silence_without_noise_is_zeros() -> None:
    clips = synthesize_silence([], n=4, length=16000, seed=0)
    assert clips.shape == (4, 16000)
    assert torch.count_nonzero(clips) == 0


# --------------------------------------------------------------------------- #
# 4. fixed-length collation
# --------------------------------------------------------------------------- #
def test_collate_pads_and_crops_to_one_second() -> None:
    collate = FixedLengthCollate(length=16000)
    batch = [
        (torch.ones(12000), 0),           # short -> pad
        (torch.ones(16000), 3),           # exact
        (torch.ones(20000), 11),          # long -> crop
    ]
    waves, labels = collate(batch)
    assert waves.shape == (3, 16000)
    assert torch.equal(labels, torch.tensor([0, 3, 11]))
    assert waves[0, 12000:].abs().sum() == 0          # padded region is zero
    assert waves[0, :12000].sum() == 12000


def test_collate_accepts_channel_dim() -> None:
    collate = FixedLengthCollate(length=16000)
    waves, _ = collate([(torch.ones(1, 16000), 0)])   # [1, L] from torchaudio
    assert waves.shape == (1, 16000)


# --------------------------------------------------------------------------- #
# 5. dataset assembly (fake items, no download)
# --------------------------------------------------------------------------- #
def fake_items():
    """(waveform, word) mimicking torchaudio SPEECHCOMMANDS output words."""
    torch.manual_seed(0)
    items = []
    for kw in KEYWORDS:                               # 4 each of 10 keywords
        for _ in range(4):
            items.append((torch.randn(16000), kw))
    for word in ("bed", "bird", "house", "wow", "three", "marvin", "tree"):
        for _ in range(40):                           # lots of unknown
            items.append((torch.randn(16000), word))
    return items


def make_dataset(split="training", augment=False, **cfg_kw):
    cfg = DataConfig(**cfg_kw)
    noise = [torch.randn(40000)]
    return SpeechCommands12(fake_items(), noise, cfg, split=split, seed=0,
                            augment=augment)


def test_augment_flag_off_by_default() -> None:
    ds = make_dataset(aug_time_shift_ms=5.0, aug_noise_prob=1.0)  # cfg on...
    assert ds._augment is None                          # ...but augment=False


def test_augment_active_only_when_enabled_and_on() -> None:
    on = make_dataset(augment=True, aug_time_shift_ms=5.0)
    off = make_dataset(augment=True, aug_time_shift_ms=0.0, aug_noise_prob=0.0)
    assert on._augment is not None                      # enabled + knob on
    assert off._augment is None                         # enabled but all-off -> no-op


def test_augment_does_not_change_labels_or_length() -> None:
    ds = make_dataset(augment=True, aug_time_shift_ms=5.0, aug_noise_prob=1.0)
    wave, label = ds[0]
    assert wave.shape[-1] == 16000 and 0 <= label < 12


def test_augment_leaves_silence_untouched() -> None:
    """Silence clips are synthesized, not real utterances -> never augmented."""
    ds = make_dataset(augment=True, aug_noise_prob=1.0, silence_fraction=0.3)
    # first silence index
    from data.speech_commands import SILENCE_INDEX
    sil = next(i for i in range(len(ds)) if ds[i][1] == SILENCE_INDEX)
    a = ds[sil][0]
    b = ds[sil][0]
    assert torch.equal(a, b)                            # deterministic (no aug)


def test_dataset_has_all_12_classes_represented() -> None:
    ds = make_dataset(silence_fraction=0.2, unknown_fraction=0.2)
    labels = [ds[i][1] for i in range(len(ds))]
    present = set(labels)
    for i in range(10):
        assert i in present, f"keyword class {i} missing"
    assert SILENCE_INDEX in present
    assert UNKNOWN_INDEX in present


def test_unknown_is_subsampled_not_dominant() -> None:
    """40*7 = 280 unknown words available, but keywords total only 40."""
    ds = make_dataset(unknown_fraction=0.1, silence_fraction=0.1)
    labels = [ds[i][1] for i in range(len(ds))]
    n_unknown = labels.count(UNKNOWN_INDEX)
    n_keyword = sum(labels.count(i) for i in range(10))
    assert n_unknown < 280                            # actually subsampled
    assert n_unknown <= n_keyword                     # not dominant


def test_dataset_items_are_fixed_length_waveforms() -> None:
    ds = make_dataset()
    wave, label = ds[0]
    assert wave.shape[-1] == 16000
    assert 0 <= label < 12


def test_dataset_length_matches_item_count() -> None:
    ds = make_dataset(silence_fraction=0.1, unknown_fraction=0.1)
    labels = [ds[i][1] for i in range(len(ds))]
    assert len(ds) == len(labels)


def test_two_splits_do_not_share_silence_seed_offsets() -> None:
    """Different splits must not synthesize the identical silence clips."""
    train = make_dataset("training", silence_fraction=0.3)
    val = make_dataset("validation", silence_fraction=0.3)
    ts = [train[i][0] for i in range(len(train)) if train[i][1] == SILENCE_INDEX]
    vs = [val[i][0] for i in range(len(val)) if val[i][1] == SILENCE_INDEX]
    assert ts and vs
    assert not torch.equal(ts[0], vs[0])


def test_end_to_end_through_afe_and_model() -> None:
    """A collated batch flows dataset -> AFE -> model -> [B, 12]."""
    from torch.utils.data import DataLoader

    from data.afe import AFEFrontend
    from models.binary_matchboxnet import BinaryMatchboxNet
    from train.config import load_config

    cfg = load_config("configs/base.yaml", {"model.C": 16, "model.T": 64,
                                             "afe.envelope_win_ms": 10.0})
    ds = make_dataset(silence_fraction=0.2, unknown_fraction=0.2)
    loader = DataLoader(ds, batch_size=8, collate_fn=FixedLengthCollate(16000))
    waves, labels = next(iter(loader))

    afe = AFEFrontend(cfg.afe)
    model = BinaryMatchboxNet(cfg.model)
    logits = model(afe(waves, target_T=cfg.model.T))
    assert logits.shape == (8, 12)
    assert labels.max() < 12
