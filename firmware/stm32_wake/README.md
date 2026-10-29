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
