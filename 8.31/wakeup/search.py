"""미분되지 않는 것들의 탐색 — tau(s) 와 START 조건.

형판 M 과 허용 오차 k 는 경사하강/좌표상승으로 맞추지만, "언제 볼 것인가"와
"언제가 시작인가"는 정수 시각이라 미분 대상이 아니다. 후보를 훑어 고른다.

회로에서 이 둘은 디코더 결선과 START detect 게이트가 되므로, 탐색 결과가
그대로 배선이다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .model import WakeupModel, detect_start, gather_states


# --------------------------------------------------------------------- AUC
def _auc(score: torch.Tensor, y: torch.Tensor) -> float:
    """순위 기반 AUC. 문턱을 정하지 않고 분리도만 잰다."""
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    r = torch.argsort(torch.argsort(score)).float() + 1.0
    n_p, n_n = pos.numel(), neg.numel()
    return ((r[y > 0.5].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)).item()


@torch.no_grad()
def offset_scores(x: torch.Tensor, start: torch.Tensor, y: torch.Tensor,
                  max_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """START 이후 각 오프셋의 단독 분리도.

    오프셋 d 마다 양성의 다수결 패턴을 형판으로 삼아 일치 개수를 세고, 그
    개수가 양성과 음성을 얼마나 가르는지를 AUC 로 잰다.

    반환 (auc [max_offset+1], proto [max_offset+1, C])
    """
    T = x.shape[2]
    aucs, protos = [], []
    pos = y > 0.5
    for d in range(max_offset + 1):
        idx = (start + d).clamp(0, T - 1)
        col = x.gather(2, idx.view(-1, 1, 1).expand(-1, x.shape[1], 1)).squeeze(2)
        p = (col[pos].mean(0) > 0.5).float() if pos.any() else torch.zeros(x.shape[1])
        cnt = (col == p.unsqueeze(0)).float().sum(1)
        aucs.append(_auc(cnt, y))
        protos.append(p)
    return torch.tensor(aucs), torch.stack(protos)


def pick_offsets(auc: torch.Tensor, n_states: int, *, min_gap: int = 3,
                 min_tau: int = 1, max_tau: Optional[int] = None) -> List[int]:
    """분리도가 높은 오프셋을 간격을 두고 고른다.

    간격을 두는 이유: 이웃 프레임은 거의 같은 정보라, 붙여 뽑으면 상태 네 개가
    사실상 하나가 된다. 회로 규모만 늘고 얻는 것이 없다.
    """
    hi = len(auc) - 1 if max_tau is None else min(max_tau, len(auc) - 1)
    order = sorted(range(min_tau, hi + 1), key=lambda d: -auc[d].item())
    chosen: List[int] = []
    for d in order:
        if len(chosen) == n_states:
            break
        if all(abs(d - c) >= min_gap for c in chosen):
            chosen.append(d)
    if len(chosen) < n_states:          # 간격을 못 지키면 조건을 풀어 채운다
        for d in order:
            if len(chosen) == n_states:
                break
            if d not in chosen:
                chosen.append(d)
    return sorted(chosen)


# ------------------------------------------------------------------ tau 탐색
@torch.no_grad()
def search_tau(model: WakeupModel, wave: torch.Tensor, y: torch.Tensor, *,
               min_gap: int = 3, refine_sweeps: int = 2,
               verbose: bool = False) -> Dict[str, object]:
    """tau(s) 를 고른다. 훑어서 후보를 잡고, 좌표상승으로 다듬는다.

    모델의 tau 버퍼와 config 를 갱신하고 요약을 돌려준다.
    """
    cfg = model.cfg
    S = cfg.head.n_states
    T = cfg.frontend.native_T

    was = model.training
    model.eval()
    x = model.features(wave)
    start, found = detect_start(x, cfg.start.k, cfg.start.min_frames)
    keep = found | (y <= 0.5)
    x, start, y_ = x[keep], start[keep], y[keep]

    # START 이후 남는 길이. timeout 은 tau[-1] 보다 커야 하므로 그만큼 여유를 둔다
    max_off = min(T - 1 - int(start.float().median().item()), cfg.head.timeout - 1)
    auc, _ = offset_scores(x, start, y_, max_off)
    tau = pick_offsets(auc, S, min_gap=min_gap, min_tau=1, max_tau=max_off)

    def score(tv: Sequence[int]) -> float:
        t = torch.tensor(sorted(tv), dtype=torch.long, device=x.device)
        xs = gather_states(x, start, t)
        model.head.init_templates(xs[y_ > 0.5])
        k = model.head.fit_k(model.head(xs)["count"], y_,
                             max_fpr=cfg.head.k_max_fpr)
        ok = (model.head(xs)["count"] >= k.unsqueeze(0)).all(dim=1)
        tpr = ok[y_ > 0.5].float().mean().item()
        fpr = ok[y_ <= 0.5].float().mean().item()
        return (1.0 + tpr) if fpr <= cfg.head.k_max_fpr else -fpr

    best = score(tau)
    for _ in range(refine_sweeps):
        moved = False
        for s in range(S):
            for cand in range(1, max_off + 1):
                if cand in tau:
                    continue
                trial = list(tau)
                trial[s] = cand
                if any(abs(a - b) < min_gap
                       for i, a in enumerate(sorted(trial))
                       for b in sorted(trial)[i + 1:]):
                    continue
                sc = score(trial)
                if sc > best + 1e-9:
                    best, tau, moved = sc, sorted(trial), True
        if not moved:
            break

    # 확정. timeout 도 함께 맞춘다 (tau[-1] 보다 크고 63 이하)
    cfg.head.tau = list(tau)
    cfg.head.timeout = min(63, max(tau[-1] + 1, cfg.head.timeout))
    model.tau = torch.tensor(tau, dtype=torch.long, device=x.device)
    cfg.head.validate(cfg.frontend)

    xs = gather_states(x, start, model.tau)
    model.head.init_templates(xs[y_ > 0.5])
    model.head.fit_k(model.head(xs)["count"], y_, max_fpr=cfg.head.k_max_fpr)
    model.train(was)

    if verbose:
        top = torch.topk(auc, min(8, auc.numel()))
        print(f"  오프셋 AUC 상위: "
              f"{[(int(i), round(float(v), 3)) for v, i in zip(*top)]}")
        print(f"  tau = {tau}, timeout = {cfg.head.timeout}, 점수 {best:.3f}")
    return {"tau": list(tau), "timeout": cfg.head.timeout,
            "auc": auc.cpu(), "score": best}


# ------------------------------------------------------------- 타이밍 합동 탐색
@torch.no_grad()
def search_timing(model: WakeupModel, wave: torch.Tensor, y: torch.Tensor,
                  candidates: Optional[Sequence[int]] = None,
                  min_frames_cands: Optional[Sequence[int]] = None,
                  min_recall: float = 0.98, min_gap: int = 3,
                  verbose: bool = False) -> Dict[str, object]:
    """START 문턱 k 와 tau(s) 를 함께 고른다.

    둘을 따로 고를 수 없다. START 가 어디서 뜨느냐가 tau 의 원점을 옮기므로,
    k 를 바꾸면 최적 tau 도 같이 바뀐다.

    k 를 지터로 고르려던 초기 시도는 틀렸다. 클립마다 발성 시작 시각 자체가
    다르므로 START 위치의 표준편차는 검출 지터가 아니라 발성 시각의 분포다.
    실제 음성에는 정답 onset 도 없다. 그래서 최종 목적함수(FPR 상한 아래에서
    TPR 최대)로 직접 고른다 -- 후보가 채널 수만큼뿐이라 그래도 싸다.

    검출률은 하드 필터로 쓴다. START 를 놓치면 그 발화는 판정 자체가 불가능해
    아무리 형판이 좋아도 되살릴 수 없기 때문이다.
    """
    C = model.cfg.frontend.n_channels
    cands = list(candidates) if candidates else list(range(1, C + 1))
    mins = list(min_frames_cands) if min_frames_cands else [1]
    was = model.training
    model.eval()
    x = model.features(wave)
    pos = y > 0.5

    rows, best = [], None
    for mf in mins:
        for k in cands:
            _, found = detect_start(x, k, mf)
            rec = found[pos].float().mean().item()
            if rec < min_recall:
                rows.append((k, mf, rec, None, None))
                continue
            model.cfg.start.k, model.cfg.start.min_frames = k, mf
            r = search_tau(model, wave, y, min_gap=min_gap, refine_sweeps=1)
            rows.append((k, mf, rec, r["tau"], r["score"]))
            if best is None or r["score"] > best[4]:
                best = (k, mf, rec, r["tau"], r["score"])

    if best is None:      # 어느 조합도 검출률을 못 맞추면 검출률 최대인 것
        k, mf, rec = max(rows, key=lambda r: r[2])[:3]
        model.cfg.start.k, model.cfg.start.min_frames = k, mf
        r = search_tau(model, wave, y, min_gap=min_gap)
        best = (k, mf, rec, r["tau"], r["score"])

    model.cfg.start.k, model.cfg.start.min_frames = best[0], best[1]
    search_tau(model, wave, y, min_gap=min_gap)      # 확정 조건으로 다시 맞춘다
    model.train(was)

    if verbose:
        for k, mf, rec, tau, sc in rows:
            mark = " <-" if (k, mf) == (best[0], best[1]) else ""
            head = f"  start_k {k:2d} x{mf}프레임: 검출률 {rec*100:5.1f}%"
            print(f"{head}  (검출률 미달, 건너뜀)" if tau is None
                  else f"{head}  tau {tau}  점수 {sc:.3f}{mark}")
    return {"k": best[0], "min_frames": best[1], "recall": best[2],
            "tau": best[3], "score": best[4], "rows": rows,
            "feasible": best[4] > 0}
