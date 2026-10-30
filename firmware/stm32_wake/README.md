# STM32 웨이크업 펌웨어 (방식 B)

STM32L476 이 AFE 비교기 16가닥을 보다가 말소리에 깨어 1 초를 모으고, FPGA 로 보낸다.
설계와 결정은 [`docs/wake_stm32_plan.md`](../../docs/wake_stm32_plan.md), 링크 규격은
[`docs/ICD.md`](../../docs/ICD.md) §7.2.

## 구조 — 판단부와 하드웨어부를 나눈다

```
core/        하드웨어를 모르는 C. Mac 에서 컴파일·검증된 그대로 보드에 들어간다
  wake_cfg.h   트리거 규칙 상수 (k=1, Δ=4, run=1, pre=30, busy=8)
  wake_core.*  칸(16비트) 하나 → 링 버퍼 · 배경 · 트리거 · 캡처 → "100칸 완성"
  link.*       CRC-8, 203 바이트 패킷, 8 바이트 결과 해석
host/        Mac 테스트용 구동기 (보드에는 안 들어감)
gen/         생성물 (커밋 안 함) -- replay_clips.h
kws_wake/    CubeIDE 프로젝트 (kws_wake.ioc). 보드부는 Core/Src/board.c,
             core/ 는 Core/Src/wake_core_link.c 가 #include 한다 (복사하지 않음)
```

**보드 확인 (2026-10-28, T470)**: SWD 는 `mode=UR` 로만 붙는다 (출하 펌웨어 탓으로 추정).
플래시 후 `board_stats.ticks` 가 3 s 에 +323 -- LSE·LPTIM1 10 ms 박자 정상. AFE 미연결이라
`last_frame` = 0xFFFE (떠 있는 입력), 배경이 그 값을 배워 트리거 없음 -- 정상.
재생 빌드는 프로젝트 설정의 심볼에 `WAKE_REPLAY` 를 더하고 `python -m export.wake_replay` 먼저.
**Stop 2 (2026-10-28)**: 박자 100 Hz, `stops == ticks` (박자마다 한 번 들어감). `BOARD_TEST=1`
(board.h, 지금 기본값)은 PB 풀다운 + DBG_STOP + 깨어날 때 3 ms 대기 -- Stop 2 중엔 DBG_STOP 을
켜도 SWD 로 RAM 이 0 으로 읽혀서다. 상태는 `tools\read_stats.ps1 [-Watch 2]` 로 본다.
풀다운을 켜니 `last_frame` 0xFFFE → 0x0000: 그동안 1 은 떠 있는 입력이었다.
전력을 잴 때와 AFE 를 붙일 때는 `BOARD_TEST=0`.

**손 펄스 (2026-10-28)**: PB0~15 를 하나씩 1.8 V 에 대면 해당 `edges[k]` 만 오른다 -- 16채널 모두
Stop 2 중 NVIC 없이 EXTI pending 으로 래치되고, 에지로는 깨지 않는다 (`stops == ticks` 유지).
위의 ⚠️ 는 이것으로 확인됐다. 손으로는 트리거가 잘 안 걸린다: 핀이 하나씩 닿는 동안의 칸이
"조용한 칸"으로 배경을 올려 문턱이 따라 오른다 (상대 트리거가 설계대로 동작).

**주입 테스트 (`read_stats.ps1 -Inject <mask>`)**: 테스트 빌드에서 칸 하나에 마스크를 OR 한다.
리셋 직후(bg 0) `0x000F`(4ch) → 트리거, 이어서 `0x0007`(3ch) → 없음, `0xFFFF` → 트리거. 트리거마다
`clips +1, timeouts +1` (FPGA 없음), 그동안 약 22 박자는 Stop 2 에 안 들어간다(전송 + 200 ms 대기).
⚠️ **배경 바닥**: `bg_q += on - (bg_q >> 5)` 는 `bg_q < 32` 에서 더 줄지 않는다. 3ch 칸 하나 뒤
bg_q = 3 으로 남아 그 뒤로 **4ch 는 다시 트리거하지 못한다** (문턱 4.09). 조용한 방에서 실효 Δ 는
4 가 아니라 4~5. `eval_trigger.py` 도 같은 식이라 계획서 수치에는 이미 반영돼 있다 -- 바꾸려면 둘 다.

**단계 3 재생 (2026-10-29): 600/600 PASS, 2회 연속.** FPGA = `kws_wake_top` + `bd_base`
(`vivado -mode batch -source rtl/build.tcl -tclargs -tag bd_base -top kws_wake_top -impl -out out/wake`,
WNS +5.29 ns), STM32 = 재생 빌드(헤드리스 빌드에 `-D WAKE_REPLAY` -- `.cproject` 를 바꾸므로 끝나면
`git checkout` 으로 되돌린다). `read_stats.ps1 -Replay` 가 SWD 로 `board_replay_go` 를 세워 시작하고
600 클립에 58.6 s (클립당 약 98 ms). 첫 시도는 600 개 전부 타임아웃이었다 -- **JP1/JP2 가 ST-LINK 쪽**이라
패킷이 PC COM4 로 갔다 (COM4 에서 `A5 64 ...` 확인). 시프터 쪽으로 옮기니 바로 통과.

**WAKE 선 (2026-10-29): `read_stats.ps1 -WakeTest` PASS.** 600/600 은 WAKE 가 끊겨도 나온다 (FPGA 는
WAKE 를 반쪽 패킷 버리기에만 쓴다). 그래서 패킷 절반을 보내고 끊은 뒤 온전한 클립을 보낸다:
WAKE 를 올리고 보내면 20/20 맞고, 안 올리면(대조군) 0/20 -- WAKE 가 FPGA 파서를 실제로 초기화한다.
`-Replay` 는 클립별 FPGA 답을 `board_replay_got[]` 에서 읽어 `out/replay/replay_<시각>.html` 을 만든다
(`tools/replay_report.py`, Vivado 동봉 Python). 골든 앞 600개는 right 396 + go 204 뿐이라 보고서의
라벨 정확도(81.3%)는 12-class test 정확도가 아니다.

**트리거 → FPGA (2026-10-29): 일반 빌드 `-Inject 0xFFFF` 10/10 `ok`.** 처음엔 3번 중 2번 실패했다.
`uart_err` 로 보니 `ErrorCode 0x08`(오버런), 전송 끝에서 57 ms -- FPGA 답이 도착한 순간이다. 결과를
폴링(`HAL_UART_Receive`)으로 받는 동안 10 ms 박자 ISR 이 끼어 바이트(87 us 간격)를 놓쳤다. 재생 빌드는
ISR 이 박자만 세서 안 드러났다. 고침: `HAL_UART_Receive_IT` 로 받고 LPTIM1 우선순위를 1 로 내려 USART2(0)
가 끼어들 수 있게 했다. 재생 600/600 과 WAKE 20/0 회귀 확인. 트리거당 깨어 있는 시간 약 9 박자.

**단계 3.5 B1 -- 스트림 주입 (2026-10-29): 두 스트림 PASS, 스트림 1 은 2회.**
`tools/make_streams.py` 가 골든 클립을 조용한 칸 사이에 깔아 `gen/stream_clips.h` 와 기대값
(`simulate_triggers`, wake_core 와 Mac 에서 비트 일치)을 만든다. 스트림 빌드(`-D WAKE_STREAM`)는 SWD 로
`board_stream_go = k` 를 받으면 박자마다 핀 대신 스트림 칸을 먹이고(시작에 `wake_init`), 걸린 클립과 FPGA
답을 `board_caps[]` 에 남긴다. `read_stats.ps1 -Stream k` -> `stream_check.py`.

| 스트림 | 내용 | 캡처 (위치·100칸 = 시뮬) | FPGA |
|---|---|---|---|
| 1 aligned (33 s) | 트리거 창이 곧 클립인 골든 13개 | 13/13 | 응답 13/13, **모델 정답 9/9** |
| 2 anywhere (19 s) | 임의 위치 8개 | 8/8 | 응답 8/8 |

골든 1000개 중 13개는 말이 클립의 30번째 칸에서 시작해 pre=30 창이 클립 그대로다 -- 그 클립들은
FPGA 가 `predictions_fixed` 를 내야 한다. 같은 스트림 안에서는 앞 단어가 남긴 배경 바닥 때문에 4개가
1~3 칸 늦게 걸린다(시뮬도 같은 위치를 예측해 캡처는 일치). 그중 하나는 1 칸 밀리자 go -> up 으로 바뀌었다
-- 트리거 위치가 정확도에 직접 닿는다는 실례. 라벨 대비는 정보용: 스트림 1 12/13, 스트림 2 7/8.
**이 테스트가 보지 않는 것**: 핀 -> EXTI 경로(손 테스트로 별도 확인), 실제 펄스 타이밍(B2).

**실제 빌드 (2026-10-29): 스트림 1 PASS -- 13/13 캡처, 모델 9/9.** 위 결과는 테스트 빌드(DBG_STOP 켬,
깨어날 때마다 3 ms 더 깨어 있음)였다. `-D WAKE_STREAM -D BOARD_TEST=0 -D BOARD_STREAM_AUTOSTART=5` 로
디버그 장치를 다 빼고, J1 전원을 껐다 켜(ST-LINK 가 켠 DBGMCU 는 전원을 내려야만 지워진다) SWD 를
건드리지 않은 채 부팅 5 s 뒤 스스로 스트림 1 을 돌렸다. 끝난 뒤 `read_stats.ps1 -Collect 1`:
`DBGMCU at boot = 0`(진짜 Stop 2), 스트림 동안 박자마다 Stop 2(stops 3722), 결과는 테스트 빌드와 같다.
덤: AFE 없이 `BOARD_TEST=0` 이면 PB 가 떠 있어 끝난 뒤에도 초당 1 개쯤 클립이 나간다(627/627 응답) --
AFE 를 붙이기 전에는 실제 빌드를 그대로 두지 말 것. 그리고 DBG_SLEEP 이 꺼진 Sleep 에서는 SWD 연결이
가끔 실패해 `read_stats.ps1` 이 다시 읽는다.

**정답 없는 캡처의 모델 답 (2026-10-29):** `experiments/stream_captures_model.py` 로 보드가 보낸 21 개
클립을 그대로 best.pt 에 넣었다 -- **FPGA == best.pt 21/21**. 라벨 대비 19/21 (틀린 둘: 1 칸 밀린
go -> up, 스트림 2 의 right -> left). FPGA 는 정답이 알려지지 않은 창에서도 모델대로 답한다.

**배경 바닥 규칙 -- 바꾸지 않는다 (2026-10-29).** 감쇠 `bg_q >> 5` 는 버림이라 bg_q < 32 에서 멈춰,
단어 몇 개 뒤 배경이 ~1 채널 남는다(위 스트림 1 의 늦은 4 개). `experiments/eval_trigger_history.py`:
GSC test 키워드 300 개를 스트림당 10 단어(사이 1~3 s 조용)로 이어 붙이고, 같은 잡음을 단어에도 섞어
비교했다. recall = 단어가 걸리고 맞은 비율, fresh = 그 단어를 처음 들었을 때(단계 1 방식).

| 잡음 | 규칙 | recall | fresh | 위치 바뀜 | 조용할 때 오경보/분 | 조용할 때 깨어 있음 |
|---|---|---|---|---|---|---|
| 없음 | 버림 (현재) | 0.783 | 0.787 | 62/300 | 0 | 0 |
| 없음 | 올림 | 0.787 | 0.787 | 0/300 | 0 | 0 |
| −40 dB | 버림 / 올림 | 0.787 / 0.780 | 0.777 | 65 / 0 | 0 | 0 |
| −30 dB | 버림 / 올림 | 0.720 / 0.717 | 0.703 | 1 / 1 | 0 | 0 |
| −20 dB | 버림 / 올림 | 0.653 / 0.643 | 0.637 / 0.620 | 1 / 2 | **1.0 / 1.8** | **2.7 % / 8.0 %** |

(반올림은 둘 사이. 300 단어라 ±2.4 pp 가 표본 잡음 -- 1 pp 차이는 의미 없다.) 결론: 바닥 때문에 조용한
방에서 단어 ~20 % 가 1~3 칸 늦게 잘리지만 **정확도 차이는 표본 잡음 안**이다. 반면 올림은 잡음
(−20 dB)에서 오경보 1.8 배, 깨어 있는 시간 3 배. −30 dB 이상에서는 잡음 자체가 바닥(~1 채널)을 만들어
규칙이 무관하다. 그래서 펌웨어 규칙은 그대로 둔다.

**단계 4 -- AFE 연결 (준비, 2026-10-29, 클립 1개 `left/54aecbd5_nohash_1`).** 핀 → EXTI → 칸 경로에
처음으로 실제 비교기를 물린다. 정답지는 소프트웨어 AFE(STFT+mel)라 비트 일치는 기준이 아니다
(`docs/edu33212a_gsc_lab.md`). 보는 것은 같은 *종류*의 그림인가 -- 채널별 발화율, 트리거 위치, 답.
`normalize: fixed` 라 임계가 절대값이므로 **재생 음량이 시험 조건의 일부**다.

```
AWG(EDU33212A, burst 1회 = 250 ms 무음 + 클립 1 s + 500 ms 무음) → AFE 입력
AFE 비교기 16가닥 → J5 → STM32 (-D WAKE_AFELOG) → J4 → FPGA (kws_wake_top, 단계 3 그대로)
```

| 순서 | 할 일 | 명령 / 확인 |
|---|---|---|
| 1 | 소프트웨어 기준 | `python -m experiments.afe_clip prep --run runs/bd_base --wav <...>/left/54aecbd5_nohash_1.wav` → `out/afe/54aecbd5_nohash_1/ref.{json,png}`: 0 dB 그림, 게인 −12~+12 dB 별 발화율·트리거 위치·답 |
| 2 | AWG 파형 | `experiments/gsc_to_edu33212a.py` (기본값이 위 burst 와 같다). **주입점 = 프리앰프 뒤 vin** (프리앰프 출력을 필터 입력에서 물리적으로 떼고, 발생기가 0.9 V 바이어스까지 준다: `--generator-zero-v 0.9`, High Z). 기준 레벨 50 mVpp = 동료 전송본의 마이크 5 mVpp × 프리앰프 10 (`analog/AFE_board/transfer_v3`), 25 / 100 mVpp 와 함께 `out/afe/<clip>/awg_vin_*mVpp/` |
| 3 | 펌웨어 | 헤드리스 빌드에 `-D WAKE_AFELOG` (BOARD_TEST 는 기본 1 그대로: 풀다운 + `edges[]`). 걸린 클립 처음 16 개와 FPGA 답·트리거 때 배경을 `board_caps[]` 에, 조용한 박자의 채널별 켜짐을 `board_quiet` 에 남긴다. Stop 2 대신 Sleep -- 언제든 SWD 로 읽힌다 |
| 4 | 무음 | AWG 출력만 연결, 신호 없음. `read_stats.ps1 -Afe -Reset` → 10 s 뒤 `read_stats.ps1 -Afe -Tag quiet`: 캡처 0 개, 무음 발화율 |
| 5 | 클립 | `-Afe -Reset` → burst 를 2 s 간격으로 5 회 → `-Afe -Tag vin50`. 25 / 100 mVpp 도 같은 방법 (`-Tag vin25`, `vin100`) |
| 6 | 비교 | `python -m experiments.afe_clip check out/afe/54aecbd5_nohash_1/ref.json out/afe/afe_vin50_*.json --plot out/afe/check_vin50.png` |

`check` 가 캡처마다 내는 것: 시뮬 캡처 대비 몇 칸 늦/이르게 걸렸나(`shift`)와 그때 1600 비트 중 같은 비율,
**발화율이 가장 닮은 소프트웨어 게인**(`like gain` -- 보드의 레벨이 학습 분포의 어디에 해당하나), FPGA 답,
같은 100칸에 대한 best.pt 답. 통과로 볼 것: burst 마다 캡처 정확히 1 개 (놓침·이중 트리거 없음), 무음에서
캡처 0, FPGA == best.pt (링크는 단계 3 에서 증명 -- 다르면 핀 경로가 아니라 다른 문제), 반복 간 `like gain`·
`shift` 가 안정. 답이 left 인지는 정보용이다: **화자 54aecbd5 는 train split** 이라(`docs/request_v4/unknown_list.csv`
의 three/54aecbd5_nohash_1, GSC 는 화자 단위로 나눈다) 모델이 이 녹음을 학습 때 봤다.

**이 클립의 소프트웨어 기준 (`prep`, 2026-10-29)**: 원래 음량(0 dB)에서 best.pt 답은 **yes** 다 -- train
클립인데도. −6 dB 이하에서만 left. 트리거는 버스트마다 1 번(−12 ~ +12 dB 모두), 캡처는 클립보다 6 칸 늦게
시작한다(0 dB). 그래서 보드가 yes 라고 답해도 그 자체는 고장이 아니다 -- `like gain` 과 함께 읽는다.

시뮬이 미리 말해 주는 것 (다른 left 클립 `28497c5b_nohash_0` 으로 도구 시험): 레벨을 올리면 단어 꼬리에서
**한 burst 에 두 번** 걸리기 시작한다 (+3 dB 이상, 둘째 창은 up 으로 답). 레벨을 −12 dB 로 내리면 답이 right
로 바뀐다. 즉 레벨 축에서 정상 구간이 있고, `like gain` 이 그 안에 드는지가 이 시험의 핵심이다.
**보지 않는 것**: J6 경로(FPGA `kws_frame_ctrl`)와의 프레임 대조 -- FPGA 비교기 경로에는 프레임을 꺼내는
수단이 없다(ILA 또는 덤프 RTL 필요). 이번엔 소프트웨어 AFE 가 기준이다.

`wake_core` 는 레지스터·인터럽트·시간을 모른다. 10 ms 마다 `wake_tick(&w, frame)` 을
부르면 `WAKE_CLIP_READY` 를 돌려줄 때가 있고, 그때 `wake_clip(&w)` 이 100칸이다.
보드 쪽(`board.c`)은 칸을 만들어 넣고, 완성되면 보내고, 답을 받으면 `wake_sent(&w)`.

## 작동 원리 (요약)

**칸 하나 (10 ms)**: `frame = 이전 박자의 GPIOB->IDR | 이번 10 ms 의 EXTI pending`.
"칸 시작 때 켜져 있었거나, 칸 동안 켜지는 순간이 있었으면 1" — FPGA `kws_frame_ctrl`
의 sticky OR, 학습의 `envelope_reduce: max` 와 같은 규칙이다.

**상태**: WARMUP(32칸: 배경만 배움) → IDLE(매 칸: 켜진 수 ≥ 배경+4 면 트리거, 아니면
배경 갱신) → CAPTURE(링 30칸 + 트리거 칸 + 69칸) → HOLD(8칸, 그리고 FPGA 답까지) → IDLE.
링 버퍼는 모든 상태에서 매 칸 채운다.

**정수만**: 배경은 `bg_q = 32 × 평균` 으로 들고 `bg_q += on − (bg_q >> 5)`.
`experiments/eval_trigger.py` 도 같은 정수식이다 — 시뮬레이터가 잰 숫자가 이 코드의 숫자다.

## Mac 에서 검증

```bash
python -m pytest tests/test_firmware_core.py tests/test_eval_trigger.py -q
```

- **`wake_core` 대 `simulate_triggers`**: 무작위 스트림 1500개 × 정책 5종(배포 정책 +
  -D 로 바꾼 4종)에서 캡처 시점·창·100칸 내용이 전부 같아야 한다. AddressSanitizer +
  UBSan 으로 빌드한다.
- 이 테스트가 버그를 실제로 잡는지 일부러 심어 확인했다 (2026-10-27): 배경을 말소리에서도
  학습, 링 한 칸 어긋남, busy 1 짧음, `≥` → `>`, 전송 후 재판정 누락 — 5종 전부 검출.
- `wake_cfg.h` 기본값 = `FIRMWARE_POLICY`, `WAKE_NATIVE` = export 의 `KWS_NATIVE_T`.
- `link.c` 의 바이트 = `export/wake_link.py`, CRC 검사값 0xF4.

## 재생 데이터 (단계 3)

```bash
python -m export.wake_replay            # → gen/replay_clips.h, bd_base 골든 600 클립 (117 KiB)
```

표준 라이브러리만 쓴다 — 노트북의 Vivado 동봉 Python 으로도 돈다.

## 보드가 오면 — CubeIDE 프로젝트 설정 (초안)

| 항목 | 설정 | 비고 |
|---|---|---|
| MCU | STM32L476RGTx (LQFP64) | |
| SYS → Debug | **Serial Wire** | PA13/PA14. PB3·PB4 가 JTAG 에서 풀려나 ch3·ch4 로 쓰인다. 안 하면 **ch4 가 늘 1** |
| RCC | LSE = Crystal (PC14/15), HSE 끔 | 32.768 kHz, J3 옆 X1 |
| 클럭 | MSI 4 MHz 로 시작 | 추론을 안 하므로 80 MHz 불필요. 115200 baud 오차 ~0.8% |
| PB0~PB15 | GPIO_EXTI, **상승 에지**, 풀 없음 | NVIC 의 EXTI0~4·EXTI9_5·EXTI15_10 은 **끔** (아래 ⚠️) |
| PA8 | GPIO 출력, 초기 Low | WAKE_FPGA |
| USART2 | PA2 TX / PA3 RX, 115200 8N1, DMA | JP1·JP2 를 **시프터 쪽**으로 (ST-LINK 쪽이면 PC 로 감) |
| LPTIM1 | 클럭 = LSE, ARR = 327, 자동 리로드 일치 인터럽트 | 328 틱 = 10.0098 ms. NVIC LPTIM1 **켬** |
| 저전력 | 대기 중 Stop 2, 깨어난 뒤 클럭 = MSI | `RCC_CFGR.STOPWUCK` |
| JP3 | **3V3** | 시프터의 FPGA 쪽 전압 = 키트 LVCMOS33 |

⚠️ **확인 필요**: EXTI 의 NVIC 를 끈 채로 pending 비트가 에지를 래치하고, 그 에지로는
Stop 2 에서 깨지 않는지 — STM32L4 레퍼런스 매뉴얼(RM0351) EXTI 장으로 확인한다. 안 되면
EXTI 인터럽트를 켜고 ISR 에서 비트만 세운다. 결과는 같고 에지마다 깨는 전력만 더 든다.

`board.c` 가 할 일:

1. LPTIM1 ISR (10 ms): `pr = EXTI->PR1 & 0xFFFF; EXTI->PR1 = pr; idr = GPIOB->IDR;`
   `frame = prev_idr | pr; prev_idr = idr;` → `wake_tick`.
2. `WAKE_CLIP_READY` 면: PA8 High → `link_frame_packet` → USART2 DMA 전송 → 결과 8 바이트
   수신 (타임아웃 200 ms) → PA8 Low → `wake_sent`.
3. 그 밖에는 Stop 2.
4. 재생 빌드(`-DWAKE_REPLAY`): `gen/replay_clips.h` 를 하나씩 보내고 `replay_expect` 와
   비교해 일치 수를 센다. 보드에 LED 가 없으므로 결과는 디버거로 변수를 보거나 FPGA 키트
   표시로 확인한다.
