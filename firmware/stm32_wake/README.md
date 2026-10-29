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
