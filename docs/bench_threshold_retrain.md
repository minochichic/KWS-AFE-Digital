# 벤치 임계 재학습 — 보드가 만들 수 있는 임계값으로 다시 학습 (2026-11-16)

## 왜

10/30 벤치(ARB 100 mVpp → AFE → STM32 → FPGA)에서 **디지털 경로는 맞다** — FPGA 답 =
`best.pt` 답 234/236, 161/163. 틀린 것은 **보드 이미지 ≠ 소프트웨어 AFE 이미지**다.
`bd_base` 의 임계값을 전압으로 옮기면(`awg_bench_20261029.md`) 회로가 낼 수 있는 범위에
7채널만 들어오고, 6채널은 900 mV 아래, 3채널은 990 mV 위를 원한다. 화면에서도 그대로
보인다: ch4·6·8·12 는 보드가 더 켜지고, ch10·13·15 는 덜 켜지고, ch2 는 세 캡처 모두 0.

그래서 회로를 모델에 맞추는 대신 **모델을 회로에 맞춘다.**

## 보드 조건 (사용자, 2026-11-16)

- 기준전압 7개: 900 / 916 / 932 / 948 / 964 / 980 / 996 mV
- 채널마다 하나를 고른다. 한 전압이 구동하는 비교기는 **최대 4개** (늘릴 수는 있음)
- 벤치: ARB 를 클립마다 100 mVpp 로 `/GIC_0/vin` 에 주입 (프리앰프 분리)

## 무엇을 만들었나

| 조각 | 하는 일 |
|---|---|
| `afe.input_pp` | 클립마다 peak-to-peak 을 고정 (ARB 와 같은 조건). 0.29 = 100 mVpp |
| `afe.threshold_grid_path` | 채널 × 전압 → raw 엔벨로프 임계 표. forward 가 임계값을 표의 한 칸으로 straight-through 로 붙인다 → **늘 납땜 가능한 비교기로 학습** |
| `afe.threshold_grid_capacity` | 전압당 채널 수 상한 (4). 최소비용 할당(헝가리안)으로 지킨다 |
| `experiments/thr_grid.py sim` | 회로 모델(DC 실측 + 스윙 시뮬 두 점)로 만든 **임시 표** |
| `experiments/thr_grid.py report` | 학습된 런이 채널마다 어느 전압을 쓰는지 → `vthr_levels.csv` |
| `experiments/afe_web.py` V_thr 칸 | 캡처마다 그때의 기준전압을 기록 |
| `experiments/thr_calib.py` | 보드 캡처로 표를 **실측 맞춤** (채널·전압마다 일치율도 낸다) |

임시 표: `analog/AFE_board/artifacts/thr_grid_sim_20261116.csv`. GSC 400클립에서 발화율은
ch0–9 가 900 mV 에서 35–48%, 996 mV 에서 8–18% 로 쓸 만하게 퍼지지만, **ch14·15 는 모든
전압에서 ≤2%** — 시뮬 스윙 곡선이 고역에서 틀렸을 가능성이 크다 (보드 화면에서는 켜진다).
그래서 보정 측정이 필요하다.

## 순서

### 1. 보정 측정 (T470, 벤치)

```powershell
python -m experiments.afe_web --open
```

1. 16채널을 모두 같은 전압(예: 900)에 놓고, 페이지의 **V_thr 칸에 `900`** 을 쓴다.
   채널마다 다르게 놓았으면 ch0..15 순서로 16개를 쉼표로.
2. ARB 를 20개 정도 바꿔 가며 재생한다 (ARB 칸도 같이 고른다).
3. 7개 전압에 대해 반복한다.
4. `out/afe/live/*.jsonl` 과 사용한 `.arb` 파일들을 보낸다.

### 2. 표 맞추기 (어디서든)

```bash
python -m experiments.thr_calib out/afe/live/*.jsonl --arb-dir <arb 폴더> \
    --sim analog/AFE_board/artifacts/thr_grid_sim_20261116.csv \
    --out analog/AFE_board/artifacts/thr_grid_board_<날짜>.csv
```

측정하지 않은 전압은 sim 표를 그 채널의 실측 칸에 맞게 배율 보정해 채운다. 최적 임계에서도
일치율이 0.9 아래인 채널은 임계 문제가 아니라 소프트웨어 필터·검출기가 그 채널과 다른
것이다 — 보드에서 먼저 확인한다.

### 3. 재학습 (원격 RTX)

```bash
python -m train.train --config configs/base.yaml --tag bd_grid --init-from runs/bd_base/best.pt \
    afe.spice_matrix_path=analog/AFE/artifacts/filterbank_matrix_board.csv \
    afe.input_pp=0.29 afe.threshold_grid_capacity=4 \
    afe.threshold_grid_path=analog/AFE_board/artifacts/thr_grid_board_<날짜>.csv
python -m experiments.thr_grid report --run runs/bd_grid
```

보정 전에 파이프라인만 먼저 돌려 보려면 표를 `thr_grid_sim_20261116.csv` 로.

격자가 정확도를 얼마나 깎는지는 같은 조건의 자유 임계 런(`bd_pp`: input_pp 만, 격자 없음)과
비교한다. `compare` 는 각 런의 임계를 그 표의 **기준전압(mV)** 으로 바꿔 보여 준다 —
자유 런이 900 아래·996 위를 원하는 채널이 곧 이 회로 범위가 깎는 채널이다.

```bash
python -m experiments.thr_grid compare --grid <표.csv> --run runs/bd_pp runs/bd_grid
```

2026-11-16 임시 표 결과: `bd_base` 0.825 → `bd_pp` 0.861 → `bd_grid_sim` 0.790.
`--init-from` 이면 input_pp 때문에 lo/hi 와 임계 초기값을 새 입력 레벨에서 다시 잰다.

### 4. 보드 반영

`vthr_levels.csv` 대로 채널별 전압을 놓고, 가중치만 export 해서 BRAM 에 싣는다
(RTL 변경 없음, CLAUDE.md §5). afe_web 으로 ARB 정답률을 소프트웨어 예측과 비교한다.

## 주의

- **이 모델은 벤치 전용이다.** `input_pp` 는 AWG 가 클립마다 음량을 맞추는 조건이다.
  실제 마이크는 음량이 그대로 들어오므로, 마이크로 갈 때는 input_pp 를 끄고 그 조건의
  표를 다시 잰다 (표 형식과 학습 경로는 그대로).
- 표는 학습과 **같은 input_pp, 같은 필터뱅크**로 만들어야 한다 (파일 머리 주석에 적힌다).
