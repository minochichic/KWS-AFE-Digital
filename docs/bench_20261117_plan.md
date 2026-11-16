# 11/17 벤치 — 할 일 (AFE 보드 · STM32 · KC705 · T470)

목표는 두 가지다.

1. **`bd_grid_sim` 의 보드 정확도** — 임시 표로 학습한 모델(소프트웨어 0.790)이 보드에서 몇이
   나오는지. 실제 회로에서 처음 재는 "7전압 격자" 숫자다.
2. **보정 측정** — 채널마다 7전압 전부에서 캡처를 모아 `thr_calib` 로 **실측 표**를 만든다.
   그 표로 원격에서 재학습한 값이 "이 회로 조건의 최적치"다.

FPGA 가중치는 **바꾸지 않는다.** FPGA 답 = `best.pt` 답이 이미 확인됐으므로(234/236, 161/163),
보드 이미지에 어떤 런을 돌려도 그게 그 가중치를 실은 FPGA 의 답이다 — 캡처만 모으면
`experiments/score_live.py` 가 나중에 어느 런으로든 채점한다. 비트스트림 재빌드는 실측 표로
재학습한 뒤 한 번만.

## 0. 오늘 밤 (준비)

원격 → T470 으로 런 하나만 가져온다 (afe_web 화면에 그 런의 답을 띄우기 위해).

```powershell
cd <T470 의 KWS-AFE-Digital>; git pull
mkdir runs\bd_grid_sim
scp "topvel_ai@TV-AI001:~/KWS-AFE-Digital/runs/bd_grid_sim/best.pt" "topvel_ai@TV-AI001:~/KWS-AFE-Digital/runs/bd_grid_sim/config.yaml" "topvel_ai@TV-AI001:~/KWS-AFE-Digital/runs/bd_grid_sim/vthr_levels.csv" runs\bd_grid_sim\
```

- [ ] T470 에 `out/afe/<ARB이름>/ref.json` 50개(10/30 것)와 동료 `.arb` 폴더가 있는지
- [ ] **ARB 20개를 미리 고른다** — 단어 10개 × 2개. 모든 설정에서 **같은 20개**를 쓴다
      (설정끼리 비교가 되고, 보정도 같은 파형으로 7전압을 덮는다)

## 1. 하드웨어 셋업 (10/29–30 과 같게)

- [ ] AWG: 오프셋 **900 mV, 100 mVpp**, `/GIC_0/vin` 주입, **프리앰프 U2301 분리**
- [ ] STM32: `-D WAKE_AFELOG` 빌드 플래시 (`firmware/stm32_wake/README.md`)
- [ ] KC705: 지금 비트스트림(`bd_base` 가중치) 그대로
- [ ] 배선: J4.2 UART_TX → J6 pin 30 (EXT27), J4.1 UART_RX ← pin 31 (EXT28),
      J4.3 WAKE → pin 32 (EXT29), GND → pin 49/50. JP1/JP2 시프터 쪽, JP3 3V3
- [ ] **기준전압 7개를 DMM 으로** — 비교기를 연결한 상태(부하 걸린 채)로 900..996 탭을 재서
      적어 둔다. 페이지 V_thr 칸에는 **공칭값(900, 916, …)** 을 쓴다 (표가 공칭값으로 묶는다).
      실측이 공칭에서 몇 mV 이상 벗어나면 그것도 알려 줄 것

## 2. ch2 확인 (10분)

납땜 수정 후 처음이다. 아래 설정 A 로 ARB 하나를 틀고 페이지 이미지에서 **ch2 줄에 1이
찍히는지** 본다. 계속 0이면 보정 전에 멈추고 원인부터 (보정 표에서 그 채널이 무의미해진다).
페이지 하단 임계 표의 IDLE(무음) 발화도 한 번 본다 — 무음에서 켜지는 채널이 있으면 기록.

## 3. 설정 A — `bd_grid_sim` 전압 (먼저, 목표 1)

| ch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mV | 980 | 948 | 916 | 948 | 996 | 932 | 996 | 948 | 980 | 932 | 932 | 900 | 932 | 900 | 900 | 900 |

전압당 채널 수: 900:4, 916:1, 932:4, 948:3, 980:2, 996:2 (964 안 씀).

```powershell
python -m experiments.afe_web --run runs/bd_grid_sim --open --vthr "980,948,916,948,996,932,996,948,980,932,932,900,932,900,900,900"
```

- 페이지에서 "재생 중인 ARB" 를 고르고 → AWG 에 그 .arb → 캡처 2–3개 → 다음 ARB.
  (ARB 를 바꾼 뒤 2초 안의 캡처는 이전 파형일 수 있어 자동으로 빠진다)
- 화면의 **best.pt 답 = `bd_grid_sim` 이 보드에서 낸 답**이다. FPGA 칸은 `bd_base` 답이라
  여기서는 보지 않는다.
- 시간이 되면 이 설정만은 20개 말고 **50개 전부** — 목표 1 의 숫자가 단단해진다.

## 4. 설정 1–7 — 보정 (목표 2)

전압당 4채널 제한 안에서 **모든 채널이 7전압을 한 번씩** 거치도록 돌린다
(설정 s 에서 채널 c 는 `LEVELS[(c + s) % 7]`, 전압당 2–3채널).
설정마다 V_thr 칸에 아래 줄을 그대로 붙여넣고, 같은 ARB 20개를 튼다.

| 설정 | V_thr (ch0..15) |
|---|---|
| 1 | `900,916,932,948,964,980,996,900,916,932,948,964,980,996,900,916` |
| 2 | `916,932,948,964,980,996,900,916,932,948,964,980,996,900,916,932` |
| 3 | `932,948,964,980,996,900,916,932,948,964,980,996,900,916,932,948` |
| 4 | `948,964,980,996,900,916,932,948,964,980,996,900,916,932,948,964` |
| 5 | `964,980,996,900,916,932,948,964,980,996,900,916,932,948,964,980` |
| 6 | `980,996,900,916,932,948,964,980,996,900,916,932,948,964,980,996` |
| 7 | `996,900,916,932,948,964,980,996,900,916,932,948,964,980,996,900` |

- **보드 점퍼를 바꾼 다음 V_thr 칸을 바꾼다** (칸을 바꾼 뒤 2초 안의 캡처는 기록에서 빠진다).
  칸과 실제 점퍼가 다르면 그 캡처는 보정을 망친다 — 설정마다 점퍼 사진 한 장 권장.
- 4채널 제한이 전기적 제한이 아니라면(한 탭에 16개를 물려도 DMM 값이 안 변하면) 16채널을
  같은 전압에 두는 7설정으로 바꿔도 된다 — 칸에 `900` 하나만 쓰면 된다. 결과는 같다.
- 시간 목표: 설정 하나 ~20–25분 → 설정 A + 1–7 ≈ 3–3.5시간.
  시간이 모자라면 **설정 1, 4, 7** 만 해도 채널마다 3전압이 실측되고 나머지는 시뮬 표를
  그 채널 실측에 맞춰 배율 보정해 채운다(`thr_calib --sim`).

## 5. 끝나고 보낼 것

- [ ] `out/afe/live/live_*.jsonl` 전부
- [ ] 이번에 튼 `.arb` 파일들 (이름 = 페이지의 ARB 이름)
- [ ] DMM 으로 잰 기준전압 7개, ch2 결과, 이상한 채널 메모

## 6. 그 다음 (내가/원격)

```bash
# 설정 A 의 보드 정확도 (bd_base, bd_grid_sim 둘 다)
python -m experiments.score_live out/afe/live/*.jsonl --run runs/bd_grid_sim runs/bd_base --levels-of runs/bd_grid_sim

# 실측 표
python -m experiments.thr_calib out/afe/live/*.jsonl --arb-dir <arb 폴더> \
    --sim analog/AFE_board/artifacts/thr_grid_sim_20261116.csv \
    --out analog/AFE_board/artifacts/thr_grid_board_20261117.csv

# 원격: 실측 표로 재학습 → 채널별 전압
python -m train.train --config configs/base.yaml --tag bd_grid_board --init-from runs/bd_base/best.pt \
    afe.spice_matrix_path=analog/AFE/artifacts/filterbank_matrix_board.csv afe.input_pp=0.29 \
    afe.threshold_grid_capacity=4 afe.threshold_grid_path=analog/AFE_board/artifacts/thr_grid_board_20261117.csv
python -m experiments.thr_grid report --run runs/bd_grid_board
```

다음 벤치: `bd_grid_board/vthr_levels.csv` 대로 점퍼 → 같은 ARB → `score_live` 로 확인 →
맞으면 `export.emit` + 비트스트림으로 FPGA 에도 싣는다.
