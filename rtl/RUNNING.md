# RTL 돌리는 법

`rtl/README.md` 가 **왜 그렇게 설계했나**라면, 여기는 **어떻게 돌리나**다.

---

## 0. 한 줄

```bash
./rtl/run_tb.sh <모듈> [태그]
```

`태그` 는 어느 export 를 쓸지이고 기본값은 `xl_g12` 다. **RTL 은 태그를 모른다** —
`rtl/*.v` 에 태그 문자열이 한 번도 안 나온다.

---

## 1. 전체 흐름

```
학습 (박스)                       runs/<태그>/best.pt + config.yaml
   │
   ├─ python -m export.emit      → rtl/gen/<태그>/  manifest.json
   │                                                parameters.vh   폭·커널·시프트
   │                                                paths.vh        ROM 경로 매크로
   │                                                *.hex           가중치·문턱값·게인
   │
   └─ python -m export.golden    → rtl/gen/<태그>/golden/
                                                    *.hex           층별 정답
                                                    paths.vh        골든 경로 매크로
                                                    predictions_fixed.txt
   │
   └─ ./rtl/run_tb.sh <모듈> <태그>
          ├─ rtl/gen/active.vh 생성 (위 셋을 include)
          ├─ verilator --lint-only
          └─ iverilog + vvp,  결과를 rtl/results.json 에 기록
```

---

## 2. 명령어

### 처음부터 (박스에서)

```bash
python -m export.emit   --tag xl_g12 --out rtl/gen/xl_g12
python -m export.golden --tag xl_g12 --clips 2 --out rtl/gen/xl_g12/golden
python -m pytest tests/ -q
./rtl/run_tb.sh top
```

### 어떤 태그가 있나

```bash
python -m experiments.list_runs
```

`export?` 열이 `yes` 인 것만 export 할 수 있다 (`config.yaml` + `best.pt` 둘 다 필요).

### 다른 트랙으로

```bash
python -m export.emit   --tag <태그> --out rtl/gen/<태그>
python -m export.golden --tag <태그> --clips 2 --out rtl/gen/<태그>/golden
./rtl/run_tb.sh top <태그>
```

**RTL 은 한 줄도 안 고친다.**

### 모듈 하나만

```bash
./rtl/run_tb.sh dw_conv          # rtl/kws_dw_conv.v + rtl/tb/tb_dw_conv.v
```

스크립트 인자는 모듈 이름에서 `kws_` 를 뗀 것이다.

### 시뮬레이터 없이 (맥에서도 됨)

```bash
python -m pytest tests/test_dw_model.py tests/test_conv1_model.py \
                 tests/test_dense_model.py tests/test_tailfmt.py \
                 tests/test_affine_rom.py tests/test_plane_model.py -q
```

이것들은 **torch 도 iverilog 도 필요 없다.** 커밋된 `.hex` 만 읽는다.

---

## 3. 무엇이 무엇을 검증하나

| 모듈 | TB 가 비교하는 것 |
|---|---|
| `bin_mac` | 생성 벡터 (`tb/vectors`) |
| `dw_conv` | `b1_s0_dw_out` + `conv2_dw_out` (dilation) |
| `pw_conv` | `b1_s0_pw_out` |
| `tcs_sub` | `b1_s0_pw_out` (dw→pw 배선) |
| `block` | `b1_add_out` (residual 정수 덧셈) |
| `plane` | `b1_add_out` — 평면이 블록을 직접 몰아서 |
| `conv1` | `conv1_out` (stride 2, int8 × ±1, 제로 패딩) |
| `affine` | `conv2_pw/conv3/conv4_out` (꼬리 epilogue ×3) |
| `dense_conv` | `conv3_acc` + `conv4_acc` (정수 MAC) |
| `tail` | `predictions_fixed.txt` (꼬리 전체) |
| `top` | `predictions_fixed.txt` (**전체 망**) |
| `frame_ctrl` | `input.hex` (비교기 배선 → 텐서) |

---

## 4. 실패했을 때 — 세 층이 서로 다른 걸 본다

| | 데이터 | 언제 | 무엇을 |
|---|---|---|---|
| **lint** | 안 봄 | 컴파일 전 | 폭 절단, 안 쓰는 신호, 래치 |
| **어서션** | 흐르는 값 | **틀린 그 사이클** | 내부 불변식 (33개) |
| **골든 벡터** | 최종 출력 | 끝나고 | 답이 다름 |

어서션은 **어디서** 틀렸는지, 골든은 **무엇이** 틀렸는지 알려준다.

### 파이썬 모델이 먼저다

`tests/test_*_model.py` 는 RTL 의 스케줄을 파이썬으로 옮겨놓은 것이다.

```
모델 통과 + RTL 실패  →  원인은 Verilog 뿐   (알고리즘·ROM·골든은 배제됨)
모델 실패             →  파형 봐야 소용없다
```

### 멈추면

어서션은 전부 **값**을 본다. 멈춤에는 볼 값이 없다. `kws_top` 의 **단계 워치독**이
어느 단계에서 몇 사이클째 안 움직이는지 찍는다.

---

## 5. 태그 메커니즘 — 왜 `rtl_fixed/` 를 안 만드나

Verilog-2005 은 매크로와 리터럴을 이어붙여 문자열을 못 만든다. 그래서 **경로 전체**가
매크로여야 한다.

| 생성물 | 누가 | 무엇 |
|---|---|---|
| `<gen>/parameters.vh` | `export.emit` | 폭·커널·시프트 |
| `<gen>/paths.vh` | `export.emit` | `` `KWS_ROM_CONV1_W `` |
| `<gen>/golden/paths.vh` | `export.golden` | `` `KWS_GOLD_INPUT `` |
| `rtl/gen/active.vh` | `run_tb.sh` | 위 셋을 include |

테스트벤치는 **`active.vh` 하나만** include 한다. 태그를 바꾸는 건 그 파일을 다시
쓰는 것이고, 그게 전부다.

> RTL 을 트랙마다 복사해야 한다면 이 구조가 실패했다는 뜻이다 — 아날로그가 바뀔
> 때마다 RTL 편집이 필요해지고, 그게 정확히 `docs/ICD.md` 가 막으려는 것이다.
> 지금은 그 주장이 **논증이 아니라 명령어**다.

---

## 6. 재-export 가 필요한 때

| 바뀐 것 | 다시 해야 할 것 |
|---|---|
| 아날로그 (frac·δ·f_c·τ·threshold) | 재학습 → `emit` + `golden` |
| 모델 가중치 (재학습) | `emit` + `golden` |
| `export/` 코드 | `emit` + `golden` |
| `rtl/*.v` | 시뮬만 |
| 채널 수 `N_CH` / `CMP_INVERT` / `FRAME_CYCLES` | **RTL 파라미터** (ICD §6 — 이 셋뿐) |

---

## 7. 설치

```bash
sudo apt install -y iverilog verilator     # Debian/Ubuntu/WSL
brew install icarus-verilog verilator      # macOS
```

verilator 가 없으면 lint 를 건너뛰고 시뮬만 돈다. **권장하지 않는다** — lint 와
시뮬은 서로 다른 걸 잡는다.
