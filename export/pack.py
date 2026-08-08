"""Bit-pack binary weights and activations for the RTL side.

Convention, fixed here once so RTL and Python cannot disagree:

* value -1 -> bit 0,  value +1 -> bit 1.  A binary dot product is then
  2*popcount(a XNOR b) - N, which is what models/binary_ops.binary_dot does.
* the packing axis is the LAST axis, LSB first: element i of the axis lands in
  bit (i % 32) of word (i // 32).
* the tail of a partial word is zero-filled, i.e. it decodes as -1. Callers
  MUST mask it off before popcount, because a zero bit is a real -1 and would
  contribute to the sum. `n_valid` is carried in the output for that reason.

LSB-first matters: Verilog indexes `word[i]` with bit 0 as LSB, so this ordering
makes `word[i]` in RTL the same element as `arr[..., i]` in Python with no
reversal anywhere. Getting this backwards is the classic bit-packing bug, so
tests/test_export.py pins it with an asymmetric pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

WORD_BITS = 32


@dataclass(frozen=True)
class PackedBits:
    """Bit-packed +-1 tensor. `words` is uint32, packed along the last axis."""

    words: torch.Tensor       # [..., ceil(n_valid / 32)] uint32
    n_valid: int              # real elements in the last axis (rest is padding)
    shape: tuple              # original tensor shape, for the manifest

    @property
    def n_words(self) -> int:
        return int(self.words.shape[-1])


def pack_pm1(x: torch.Tensor) -> PackedBits:
    """Pack a +-1 tensor along its last axis, LSB first, -1 -> 0 and +1 -> 1."""
    uniq = set(torch.unique(x).tolist())
    if not uniq <= {-1.0, 1.0, -1, 1}:
        raise ValueError(f"expected a +-1 tensor, saw values {sorted(uniq)}")

    bits = (x > 0).to(torch.int64)
    n = int(bits.shape[-1])
    n_words = (n + WORD_BITS - 1) // WORD_BITS
    pad = n_words * WORD_BITS - n
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))          # zeros = -1

    bits = bits.reshape(*bits.shape[:-1], n_words, WORD_BITS)
    weight = (1 << torch.arange(WORD_BITS, dtype=torch.int64,
                                device=bits.device))            # LSB first
    words = (bits * weight).sum(dim=-1)
    return PackedBits(words=words.to(torch.uint32), n_valid=n, shape=tuple(x.shape))


def unpack_pm1(p: PackedBits) -> torch.Tensor:
    """Inverse of pack_pm1, dropping the zero padding."""
    w = p.words.to(torch.int64)
    shifts = torch.arange(WORD_BITS, dtype=torch.int64, device=w.device)
    bits = (w.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(*w.shape[:-1], -1)[..., :p.n_valid]
    return torch.where(bits > 0, 1.0, -1.0)


def to_hex_words(p: PackedBits) -> list:
    """Flat list of 8-digit hex strings, for a Verilog $readmemh ROM."""
    return [f"{int(v):08x}" for v in p.words.reshape(-1).tolist()]


def quantize_int8(w: torch.Tensor) -> tuple:
    """Symmetric per-output-channel int8 quantization: returns (q, scale).

    w ~= q * scale, with q in [-127, 127] and one positive scale per output
    channel. 127 rather than 128 keeps the range symmetric so that negating a
    weight cannot overflow -- an asymmetric range is a standard source of
    off-by-one mismatches between the Python reference and RTL.
    """
    axes = tuple(range(1, w.dim()))
    amax = w.detach().abs().amax(dim=axes, keepdim=True).clamp_min(1e-12)
    scale = (amax / 127.0)
    q = torch.round(w.detach() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.reshape(-1)
