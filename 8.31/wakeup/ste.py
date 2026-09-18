"""Straight-through estimator.

비교와 계단은 미분이 0이라 그대로는 학습이 되지 않는다. 순방향은 계단 그대로
두고, 역방향만 |x| <= clip 구간에서 항등으로 통과시킨다 (Hubara/Courbariaux).
본 과제의 채널 문턱 학습에서 쓰던 것과 같은 기법이다.
"""
from __future__ import annotations

import torch


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.clip = clip
        # torch.sign(0) == 0 은 유효한 이진값이 아니다 -> 0 은 +1 로 보낸다
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_out * (x.abs() <= ctx.clip).to(grad_out.dtype), None


def sign_ste(x: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """{-1,+1} 로 이진화."""
    return _SignSTE.apply(x, clip)


def step_ste(x: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """{0,1} 로 이진화. 비교기 한 개에 해당한다."""
    return (sign_ste(x, clip) + 1.0) * 0.5


def gate_ste(logit: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """{0,1} 게이트. 0 이면 그 채널은 X(저항 미연결)가 된다."""
    return step_ste(logit, clip)
