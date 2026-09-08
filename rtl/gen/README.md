# `rtl/gen/` — 생성물이지만 **커밋한다**

여기 있는 파일은 `export/`가 학습된 체크포인트에서 뽑아낸 것이다. 생성에는
**torch와 체크포인트**가 필요하고 그건 학습 박스에만 있는데, **테스트벤치는
iverilog가 도는 아무 데서나 필요하다.** 그래서 결과물이 레포에 같이 다닌다.

`runs/`는 `.gitignore`에 있다 (체크포인트가 크다). 그러니 커밋할 사본은
반드시 여기로 뽑는다.

## 다시 뽑기

```bash
cd "$(git rev-parse --show-toplevel)"
python -m export.emit   --tag xl_g12 --out rtl/gen/xl_g12
python -m export.golden --tag xl_g12 --out rtl/gen/xl_g12/golden --clips 2
```

**재학습했으면 반드시 다시 뽑는다.** 안 그러면 테스트벤치가 옛 가중치로
새 RTL을 검증한다 — 통과해도 아무 의미가 없다.

## 뽑은 뒤 커밋 누락 점검

`.hex`만 눈에 띄어서 **`paths.vh` 두 개를 빠뜨리기 쉽다.** 실제로 한 번 빠졌고,
그 상태로 새로 클론하면 `run_tb.sh`도 `rtl/build.tcl`도 시작조차 못 한다
(2026-09-08 에 매니페스트로 복구).

```bash
git ls-files rtl/gen/<tag>/paths.vh rtl/gen/<tag>/golden/paths.vh
```

두 줄이 다 나와야 한다. 하나라도 비면 커밋이 안 된 것이다.

`paths.vh`는 `manifest.json`/`golden.json`에서 기계적으로 재생성되므로, 잃어도
체크포인트 없이 되살릴 수 있다 — 다만 그걸 알아채는 시점이 보통 "다른 기계에서
클론했는데 안 돌 때"라 값이 비싸다.

## 클립 수를 2로 두는 이유

기능 검증에는 2개면 충분하고, 8개면 누산기 덤프가 4배가 된다. 더 넓은 회귀가
필요하면 `runs/<tag>/rtl/golden`에 8개 이상으로 뽑아 쓰고, 커밋은 하지 않는다.

## 들어 있는 것

| | |
|---|---|
| `parameters.vh` | 치수·비트폭. RTL이 `include` 한다 |
| `*_w.hex` | 가중치 (이진은 비트팩킹, int8은 2의 보수) |
| `*_t.hex` | 융합된 정수 threshold + 극성 |
| `manifest.json` | 층 목록·epilogue·경계. `parameters.vh`의 출처 |
| `golden/*_acc.hex` | 층별 정수 누산기 |
| `golden/*_out.hex` | 층별 ±1 출력 (비트팩킹) |
| `golden/logits.txt` | 최종 로짓 · `predictions.txt` |

포맷 규약은 `export/pack.py` 상단과 `docs/ICD.md` §5에 있다.
