# 칩 자체 검사(self-test) 현황 정리 — 2026-09-17

> 이번 주 목표: **GSC 클립 수백~수천 개를 XC7S75 칩 위의 우리 RTL 에 넣고, 칩이
> 파이썬 모델과 같은 답을 내는지 확인한다.** AFE 보드 연결은 이번 범위가 아니다.
>
> 브랜치 `codex/windowing` (워크트리 `KWS-AFE-Digital-windowing`).
> 관련 커밋: `24e5466`(벡터) → `76da70f`(하네스) → `64e9e42`(build -selftest) → `a103b3a`(VIO).

---

## 0. 한눈에 보기

| 단계 | 상태 | 핵심 수치 |
|---|---|---|
| 벡터 export (원격 GPU) | ✅ 완료 | 2 모델 × 1000 클립, test split |
| RTL 시뮬 (XSim, 노트북) | ✅ PASS | 2 클립 2/2 일치, **57.11 ms/클립** |
| 합성 (600 클립 ROM 포함) | ✅ 완료 | 18 분, 에러 0 / 크리티컬 0 |
| 배치배선 + 비트스트림 | ✅ 완료 | **WNS +4.96 ns**, WHS +0.006 ns, DRC 에러 0 |
| 보드 실행 (`bd_base`, 클립 0–599) | ✅ **PASS** | **600 / 600 일치 × 2 회**, 57.5 ms/클립 (§5.3) |
| 모델 2 (`bd_base_ft20_partial75`) | ⏳ 벡터만 준비 | §8 |

---

## 1. 두 가지 질문

1. **`bd_base`** (동료 보드 필터뱅크, 트랙 2, 전체 test 정확도 **0.825** —
   `docs/ABLATIONS.md`): 칩이 클립마다 파이썬과 **같은 클래스**를 내는가?
2. **`bd_base_ft20_partial75`** (always-on 스트리밍용 파인튜닝 모델): 같은 질문.

"같은 클래스" 의 정답지는 **정답 라벨이 아니라** 파이썬 정수(fixed-point) 경로의
예측 `predictions_fixed.txt` 다. 이유는 §4.

---

## 2. 환경

| 역할 | 장비 | 도구 |
|---|---|---|
| 학습 / 벡터 export | 원격 GPU 박스 `topvel_ai@TV-AI001:~/KWS-AFE-Digital` (`codex/windowing`) | PyTorch, `export/golden.py` |
| 벡터 가공 / 시뮬 / 합성 / 구현 | 노트북 ThinkPad T470 (i5-7300U 2코어, RAM 16 GB, Win10) | **Vivado / XSim 2026.1** (무료 ML Standard — Spartan-7 은 라이선스 불필요) |
| 파이썬 (노트북) | Vivado 동봉 Python 3.13 (`C:\AMDDesignTools\2026.1\tps\win64\python-3.13.0`) | 표준 라이브러리만 (torch 없음) |
| 타깃 | Hanback 키트, **XC7S75-FGGA484-1** | JTAG (DLC10 → J4) |

Vivado 흐름은 **non-project 배치**(`create_project -in_memory`)다. `.xpr` 없음.
GUI 는 Hardware Manager / 체크포인트 열람에만 쓴다.

---

## 3. 데이터: 어떤 클립을 썼나

### 3.1 export (원격)

```bash
python -m export.golden --tag bd_base --split test --clips 1000 --vectors-only --out rtl/gen/bd_base/selftest
python -m export.golden --tag bd_base_ft20_partial75 --split test --clips 1000 --vectors-only --out rtl/gen/bd_base_ft20_partial75/selftest
```

- Google Speech Commands v2, **test split 의 앞 1000 클립** (로더 순서 그대로).
- 각 클립 = AFE 소프트웨어 시뮬 → **`[16, 128]` 이진 이미지** → 프레임당 16 비트 워드.
- `--vectors-only`: 층별 중간값 없이 입력/pooled/최종 예측만. (불일치가 나오면 그
  클립만 층별로 다시 뽑는다.)

산출물 (`rtl/gen/<tag>/selftest/`, 커밋됨):

| 파일 | 내용 |
|---|---|
| `input.hex` | 128,000 줄 = 1000 × 128 프레임 |
| `predictions_fixed.txt` | 파이썬 **정수 경로** 예측 (10 진수) — 칩의 정답지 |
| `predictions.txt` | 파이썬 float 경로 예측 (참고) |
| `labels.txt` | 정답 라벨 |
| `logits.txt`, `pooled.txt` | 디버깅용 |
| `expected.hex` | `predictions_fixed` 를 16 진수로 (`export/predictions_to_hex.py`) |

`expected.hex` 가 따로 필요한 이유: `$readmemh` 는 16 진수로 읽는다. 10 진수
`11`(unknown)을 그대로 넣으면 `0x11 = 17` 로 읽혀 틀린다.

### 3.2 슬라이스 (노트북)

```bash
python -m export.slice_selftest rtl/gen/bd_base/selftest --clips 600 --base 0 --out out/selftest/bd_base
```

- 칩 BRAM 예산 때문에 **600 개** (클립 0–599). 32 비트로 잡았을 때 상한이 약 680.
- `$readmemh` 의 범위 인자는 파일 오프셋이 아니라 메모리 주소라서, 오프셋은 파일을
  잘라서 해결한다(`--base`).
- `paths.vh` 에 `KWS_ST_CLIPS/BASE/T_IN/CLIP_FILE/EXP_FILE` 을 적어 시뮬과 합성이
  같은 값을 읽는다.

### 3.3 ⚠️ 클래스 편중 — 반드시 알고 있을 것

test 로더가 **섞이지 않은 순서**라 앞 1000 개가 클래스별로 몰려 있다.

| 구간 | 라벨 분포 |
|---|---|
| 클립 0–599 (지금 칩에 올라간 것) | `right`(5) 396 개, `go`(9) 204 개 — **2 클래스뿐** |
| 클립 600–999 | `no`(1) 202 개, `go`(9) 198 개 |

**의미:** 이 검사는 "데이터패스가 정수 연산을 정확히 재현하는가" 를 보는 것이라
클래스가 적어도 목적은 유효하다 — 입력 이미지는 600 개 전부 다르고, 네트워크
전체(conv1~conv4, 모든 가중치)를 매번 지난다. 다만 **12 클래스 전부의 출력 경로를
본 것은 아니다.** 다만 **예측값**(`predictions_fixed`, 오분류 포함)은 클립 0–599 에서
0 을 뺀 11 개 클래스가 나온다 — 5:329, 9:166, 11:32, 1:23, 3:17, 4:17, 6:10, 2:3,
7·8·10 각 1. 출력단 커버리지는 라벨 분포보다 넓지만 한두 개짜리 클래스가 많다. 다음 비트스트림은
**섞인/층화된 클립**으로 뽑는 것이 맞다 (§8).

### 3.4 파이썬 기준 정확도 (이 클립들에서)

| 모델 | fixed, 600 개 | fixed, 1000 개 | float, 1000 개 | fixed == float (1000 중) |
|---|---|---|---|---|
| `bd_base` | **0.813** | 0.809 | 0.811 | 996 |
| `bd_base_ft20_partial75` | **0.818** | 0.808 | 0.807 | 997 |

전체 test 셋 정확도(0.825)와 조금 다른 것은 위 편중 때문이다. fixed 와 float 이
4 개 다른 것은 양자화의 정상적인 결과다.

---

## 4. 평가 기준

**칩 출력 == `predictions_fixed.txt`, 클립마다, 전부.**

- 파이썬 정수 경로와 RTL 은 같은 정수 연산이다. 근사가 없으므로 **하나라도 다르면
  버그**다. 허용 오차 개념이 없다.
- float 예측(`predictions.txt`)과 비교하면 안 된다 — 양자화 때문에 원래 4 개쯤 다르다.
- 전부 일치하면 **칩 정확도 = 파이썬 fixed 정확도**(600 개 기준 0.813)가 된다.
  정확도는 칩에서 재는 게 아니라 이 동치로 따라온다.
- 추가로 **두 번 돌려 같은 값**인지 본다(결정성). 타이밍 위반·미초기화 레지스터는
  실행마다 다르게 틀린다.

채점은 **칩 안에서** 한다 (§5.1). 칩 밖으로 나오는 것은 카운터뿐이다.

---

## 5. 하드웨어 구성

### 5.1 계층

```
kws_selftest_board          핀: clk(B6), rst_n(EXT16) 두 개뿐
 ├─ vio_st (VIO IP)         JTAG 으로 결과 읽기 + go / soft_rst 주기
 └─ kws_selftest_top        ← 시뮬과 칩이 공유하는 모듈
     ├─ kws_selftest        클립 ROM(BRAM) + 정답 ROM + FSM + 채점 카운터
     └─ kws_top_synth       네트워크 (가중치 $readmemh)
```

- **`kws_selftest`**: 클립 ROM 에서 프레임을 네트워크의 ready 속도로 밀어 넣고
  (ready/valid), 클래스가 나오면 정답 ROM 과 비교해 `total / match / any_fail /
  first_fail_idx / got / exp` 를 갱신한다. 실시간 10 ms 프레임 타이머가 없어서
  클립당 약 57 ms 로 끝난다.
- **VIO**: 결과 약 58 비트를 핀 없이 이미 꽂힌 JTAG 케이블로 읽는다. LED/FND,
  추가 배선 없음. J6(AFE 커넥터)은 구동하지 않으므로 뭐가 꽂혀 있어도 안전.
- **go** 는 VIO 값 0→1 에지에서 한 번만 시작. 두 번째 실행 전에는 soft_rst 로 지운다
  (`any_fail/first_fail_*` 는 go 로 안 지워진다 — 첫 실패 보존용).

### 5.2 실행 방법

```bash
cd C:\Users\okbong\repos\KWS-AFE-Digital-windowing
vivado -mode batch -source rtl/selftest/run_selftest.tcl -nolog -nojournal -notrace
```

`run_selftest.tcl` 이 하는 일: 프로그램 → (soft_rst → go → 2 초마다 `total/match`
폴링 → done) × 2 → 요약. 약 2 분.

| 결과 | 판정 |
|---|---|
| 두 번 모두 `total 600 match 600 any_fail 0`, 마지막 줄 `PASS` | 성공 |
| `FIRST MISMATCH: clip N got X want Y` | 불일치 — 클립 N 을 층별 벡터로 다시 export 해 시뮬 |
| 두 실행 값이 다름 | 결정성 실패 — 타이밍/리셋 의심 |

### 5.3 첫 보드 실행 결과 (2026-09-17 16:46, `-runs 1 -noprog`)

```
== run 1 / 1 ==
     2.0 s  total   35  match   35  any_fail 0
    ...
    34.5 s  total  600  match  600  any_fail 0
   34.5 s for 600 clips = 57.5 ms/clip (poll resolution 2 s)
== summary ==
total 600  match 600  any_fail 0
PASS chip reproduces predictions_fixed.txt on every clip
```

- **XC7S75 위의 RTL 이 `bd_base` 파이썬 정수 경로를 600 클립 전부 재현했다.**
  따라서 이 600 개에 대한 칩 정확도 = 0.813 (§3.4).
- 속도 57.5 ms/클립 은 시뮬 57.11 ms 와 일치한다(폴링 해상도 2 s 오차 안).
- 처음 시도는 `debug hub core was not detected` 로 실패했다. 원인은 베이스 보드
  클럭 선택 스위치가 **0 Hz** 였던 것 — `docs/hanback_kit.md` §3.2. 스위치를
  50 MHz 로 돌리고 재프로그래밍 없이 다시 읽어 통과.
- **결정성 확인 (18:34, `-runs 2`, 보드 전원 재투입 후 재프로그래밍):** 두 실행 모두
  `600 / 600, any_fail 0`, 57.6 / 57.9 ms/클립. 요약 `2 run(s) identical`.
  전원 사이클 → 새 프로그래밍 → soft_rst → 실행을 거쳐도 같은 결과이므로,
  초기화 안 된 상태나 타이밍 경계에 기대는 동작이 아니다.

  비교 대상은 **최종 카운터**(`done/total/match/any_fail/first_fail_*`)다. 전부
  일치했으므로 클립별 결과가 같다는 뜻이기도 하다 — 두 실행 모두 정답지와 600 개
  전부 같았기 때문이다.

---

## 6. 시뮬레이션 (XSim)

```powershell
python -m export.slice_selftest rtl/gen/bd_base/selftest --clips 2 --out out/selftest/bd_base_sim
powershell -NoProfile -ExecutionPolicy Bypass -File .\rtl\run_xsim.ps1 -Name selftest -Tag bd_base -Vectors out/selftest/bd_base_sim
```

- 테스트벤치 `rtl/tb/tb_selftest.v`, 50 MHz 클럭(보드와 동일).
- **2 클립만** 돌린다. 클립 하나가 약 2.86 M 사이클이라 600 개는 시뮬로 며칠이다.
  여기서 보는 것은 정확도가 아니라 **하네스가 프레임을 맞게 먹이는가**
  (ready/valid, start 펄스, 클래스 래치, 카운터, 16 진수 해석). 그게 칩이 존재하는 이유다.

결과 (`out/xsim/selftest/xsim.log`, 2026-09-17 00:43):

```
== tb_selftest: 2 clips, T_IN 128 ==
  clip 0 done at 57113090000: match=1
  clip 1 done at 114225930000: match=2
PASS all 2 clips reproduce predictions_fixed.txt
2 clips checked, 0 failures
```

→ **클립당 57.11 ms** (시뮬 시간). 칩에서 600 개 ≈ **34 초** 예상.

하네스 개발 중 시뮬이 잡은 버그 1 건: 처음엔 `net_valid` 를 레지스터로 냈다가
`kws_top.v` 의 assert(`in_valid && !in_ready`)에 걸렸다. ROM 을 동기 읽기로 두고
`net_valid = (st == S_HOLD) && net_ready` 조합으로 바꿔 해결.

그보다 앞서(다른 PC, iverilog/yosys) 네트워크 자체는 층별 골든 벡터로 검증되어
있었다(`rtl/run_tb.sh`, `tb_top` 등). 이번 작업은 그 위에 **대량 클립 경로**를 얹은 것이다.

---

## 7. 합성 · 구현 결과 (`kws_selftest_board`, 600 클립)

```bash
vivado -mode batch -source rtl/build.tcl -notrace -tclargs -tag bd_base -top kws_selftest_board -selftest out/selftest/bd_base -impl
```

산출물: `out/synth/xc7s75fgga484-1/` — `kws_selftest_board.bit / .ltx`,
`post_synth.dcp`, `post_route.dcp`, `utilization*.rpt`, `timing_impl.rpt`,
`power_impl.rpt`, `worst_path.rpt`.

소요 시간 (노트북): 합성 **약 18 분**, opt+place+route+bitstream 약 10 분.
합성 시간은 대부분 "Cross Boundary and Area Optimization" 단계에서 쓰였다.

### 7.1 면적 (Area)

| 자원 | 사용 | 전체 | % |
|---|---|---|---|
| Slice LUT | 20,731 | 48,000 | **43.2** |
|   LUT as Logic | 20,009 | | 41.7 |
|   LUT as Memory (분산 RAM + SRL) | 722 | 17,600 | 4.1 |
| Slice Register (FF) | 17,839 | 96,000 | 18.6 |
| F7 / F8 Mux | 4,318 / 817 | | 13.5 / 5.1 |
| Block RAM Tile | **51** (RAMB36 48 + RAMB18 6) | 90 | **56.7** |
| DSP48E1 | 7 | 140 | 5.0 |
| I/O | 2 | 338 | 0.6 |

모듈별 (합성 후, `utilization.rpt`):

| 인스턴스 | LUT | FF | RAMB36 | RAMB18 | DSP |
|---|---|---|---|---|---|
| `u_net` 네트워크 | 20,047 | 16,425 | 0 | 6 | 7 |
| `u_st` 하네스 + 클립 ROM | **153** | 111 | **48** | 0 | 0 |
| VIO + dbg_hub | 나머지 약 500 | | | | |

읽는 법:

- **클립 ROM 은 BRAM 으로 들어갔다** (RAMB36 48 개). LUT 로 샜다면 수만 LUT 가 늘었을 것이다.
  네트워크 LUT 는 네트워크 단독 구현(20,148)과 같다.
- 예상은 RAMB18 134 개였는데 실제로 더 적은 이유: 워드를 32 비트로 선언했지만 상위
  16 비트가 쓰이지 않아 Vivado 가 잘랐다(`Synth 8-3936`). 실제 ROM = 600×128×16 =
  1,229 Kbit. → **16 비트 기준이면 1000 클립도 한 비트스트림에 들어갈 여유**가 있다.
- 네트워크 가중치 ROM 대부분은 비동기 읽기라 BRAM 이 아니라 LUT(RAM64M, MUXF7/8)로
  구현돼 있다. LUT 42 % 의 주원인이고, 다른 모델/보드로 키울 때 첫 최적화 후보다.

### 7.2 타이밍 (Timing) — 50 MHz (20 ns)

| 항목 | 값 |
|---|---|
| Setup WNS | **+4.960 ns** (위반 0 / 55,004 endpoint) |
| Hold WHS | **+0.006 ns** (위반 0) |
| Pulse width WPWS | +8.750 ns |
| 결론 | `All user specified timing constraints are met.` |
| 대략 Fmax | 1 / (20 − 4.96) ≈ **66 MHz** |

가장 느린 경로 (`worst_path.rpt`):

```
u_top/u_net/u_top/u_c2/valid_reg[26]  →  u_c2/u_mac/acc_reg[3]
데이터 지연 14.99 ns = 로직 4.01 ns (27 %) + 배선 10.99 ns (73 %)
로직 단수 16 (CARRY4 1, LUT2 2, LUT3 2, LUT4 3, LUT5 4, LUT6 4)
```

conv2 depthwise(k=29, dilation 2)의 MAC 입력 선택 → 누산기 경로다. 배선이 73 % 라
배치 밀도에 민감하다. -1(가장 느린) 등급에서 통과했으므로 여유는 충분하다.
배선 중간에 `WHS=-0.207` 이 찍혔던 것은 hold-fix 단계 전 값이고 최종은 양수다.

### 7.3 전력 (Power) — ⚠️ 추정치

`report_power` (vectorless, 신뢰도 **Medium**, 주변 25 °C):

| 항목 | W |
|---|---|
| **Total on-chip** | **0.184** |
| Device static (누설) | 0.094 |
| Dynamic | 0.090 |
|   Block RAM | 0.029 |
|   Slice logic | 0.022 |
|   Signals (배선) | 0.021 |
|   Clocks | 0.016 |
|   DSP | 0.002 |
| Junction 온도 | 25.5 °C |

해석 주의:

- **실측이 아니다.** 스위칭 확률을 도구 기본값으로 가정한 추정이다. SAIF(시뮬 활동
  파일)를 넣으면 신뢰도가 올라간다.
- 하네스·클립 ROM·VIO 가 포함된 값이다. BRAM 0.029 W 의 대부분은 클립 ROM 이다.
- 클립당 에너지로 환산하면 0.184 W × 57.1 ms ≈ **10.5 mJ**(dynamic 만이면 ≈ 5.1 mJ).
  단 이 "57 ms" 는 ready 속도로 몰아 돌린 시간이고, 실제 always-on 동작은 10 Hz
  추론이라 대부분 대기다. CLAUDE.md 의 15 mJ 목표는 **연산 카운트 기반 추정**으로
  정의돼 있으므로 이 수치와 직접 비교하지 않는다.
- 이 칩(Spartan-7)은 정적 전력(0.094 W)이 동적과 비슷하다 — FPGA 프로토타입의 한계이고,
  ASIC/저전력 MCU 수치와는 다른 세계다.

### 7.4 경고 요약

전체 117 경고, 크리티컬 0.

| 코드 | 개수 | 내용 | 조치 |
|---|---|---|---|
| Synth 8-7129 | 100 | VIO IP 내부 미사용 포트 | 무시 (IP 고유) |
| Synth 8-693 / 8-3848 | 6 / 6 | pw_conv 의 T_FILE 없는 인스턴스 | 설계상 의도 |
| Synth 8-6014 | 3 | assert 용 레지스터 제거 | 무시 |
| Synth 8-3936 | 1 | 클립 ROM 워드 32→16 비트 | 의도 (§7.1) |
| DRC CFGBVS-1 | 1 | CFGBVS/CONFIG_VOLTAGE 미지정 | **미결** — J4 VREF 측정 후 지정 |

---

## 8. 남은 일

1. ~~보드 실행~~ ✅ 600/600 PASS, 2 회 동일 — 전원 재투입 후에도 (§5.3).
2. **클래스 커버리지 보강** — 클립 선택을 섞거나 클래스별로 층화해서 다시 export.
   또는 워드 폭을 16 비트로 선언해 1000 클립을 한 비트스트림에.
3. **오프셋 스윕** — `--base 600 --clips 400` 등으로 나머지 구간, 수천 개로 확장.
4. **질문 2** — `bd_base_ft20_partial75` 로 같은 흐름
   (`slice_selftest` → `build.tcl -tag bd_base_ft20_partial75 ... -impl` → `run_selftest.tcl`).
   이 모델의 투표/창(window) 로직은 네트워크 뒤의 작은 로직이라 별도 짧은 TB 로 검증.
5. **CFGBVS** — J4 VREF 측정 후 xdc 에 지정해 경고 제거.
6. (선택) 전력 신뢰도 — XSim 에서 SAIF 를 뽑아 `read_saif` 후 `report_power`.
