"""The offset measurement's fill beds -- the part that produced a wrong answer.

The first version filled the window with WHITE noise at the clip's floor, and
that cost 15pp against a silent fill, dwarfing the position effect it was
supposed to isolate. A relative threshold divides by the cross-channel max, and
a flat spectrum lifts that denominator in every band at once. So the beds are
now level-matched but spectrally distinct, and these tests pin the properties
the comparison depends on.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from experiments.window_offset import (FILLS, make_bed, noise_floor_rms,
                                       reposition, word_span)

SR = 16000


def _clip(n=4, a=4000, b=9000, floor=0.002):
    x = torch.randn(n, SR) * floor
    x[:, a:b] += torch.randn(n, b - a)
    return x


def test_beds_are_level_matched_so_only_spectrum_differs() -> None:
    """If the beds differed in level too, any gap between them would be
    confounded -- the whole point is to compare spectral shape."""
    x = _clip()
    a, b = word_span(x)
    floor = noise_floor_rms(x, a, b)
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(32, SR) * 0.05          # stand-in for room crops
    for kind in ("room", "white"):
        bed = make_bed(kind, x, floor, bank, g)
        assert torch.allclose(bed.std(1), floor, rtol=1e-3), kind


def test_zero_bed_is_exactly_zero() -> None:
    """Not 'small' -- xmix/xlse subtract d, so an exactly-silent frame has a
    negative numerator and is decisively off."""
    x = _clip()
    a, b = word_span(x)
    bed = make_bed("zero", x, noise_floor_rms(x, a, b), None, torch.Generator())
    assert torch.count_nonzero(bed) == 0


def test_room_bed_keeps_the_bank_spectrum() -> None:
    """Scaling must not whiten the crop: a low-frequency bank has to stay
    low-frequency, or 'room' silently becomes 'white' again."""
    x = _clip()
    a, b = word_span(x)
    t = torch.arange(SR, dtype=torch.float32) / SR
    bank = torch.sin(2 * torch.pi * 100.0 * t).expand(8, SR).contiguous()
    bed = make_bed("room", x, noise_floor_rms(x, a, b), bank, torch.Generator())
    # a 100 Hz tone barely moves sample to sample; white noise does
    rough = (bed.diff(dim=1).std(1) / bed.std(1)).mean()
    assert float(rough) < 0.2


def test_room_without_a_bank_fails_loudly() -> None:
    """Falling back to white would answer a different question under the same
    column heading."""
    x = _clip()
    a, b = word_span(x)
    with pytest.raises(RuntimeError, match="_background_noise_"):
        make_bed("room", x, noise_floor_rms(x, a, b), None, torch.Generator())


def test_unknown_fill_is_rejected() -> None:
    x = _clip()
    a, b = word_span(x)
    with pytest.raises(ValueError, match="noise"):   # the old name is gone
        make_bed("noise", x, noise_floor_rms(x, a, b), None, torch.Generator())
    assert set(FILLS) == {"room", "zero", "white"}


def test_reposition_conserves_energy_at_both_edges() -> None:
    """The measurement's core promise: the word is MOVED, never cut."""
    x = _clip()
    a, b = word_span(x)
    span = (x.shape[1] - (b - a)).clamp_min(0)
    e0 = [float(x[i, a[i]:b[i]].pow(2).sum()) for i in range(x.shape[0])]
    for p in (0.0, 0.5, 1.0):
        y = reposition(x, a, b, (span.float() * p).long() - a,
                       torch.zeros_like(x))
        for i in range(x.shape[0]):
            assert abs(float(y[i].pow(2).sum()) / e0[i] - 1.0) < 1e-4
