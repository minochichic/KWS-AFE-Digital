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
