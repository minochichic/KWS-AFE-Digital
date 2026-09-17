# Always-on 칩 검증 계획 (2026-09-17 결정)

## 목표

**always-on 경로 전체(프레임 포착 → sliding window → 네트워크 → 정수 margin
투표 → cooldown)가 칩에서 Python 정수 기준과 케이스마다 똑같이 나오는가.**

클립 자체 검사(`docs/selftest_report.md`)와 같은 논리다. 칩 결과가 Python 정수
결과와 전부 같으면 칩의 스트리밍 recall·오검출은 Python 정수 수치와 같다.

## 결정 사항

| 항목 | 결정 |
|---|---|
| test split | **연다.** 정수 정책을 val 로 확인한 뒤 test 1 회 평가, 칩 벡터도 test 에서 |
| 칩 케이스 수 | **240** (클래스당 20, 교차 순서) |
| 모델 / 정책 | `bd_base_ft20_partial75`, hop 10, N=5, cooldown 10, `MARGIN_INT=4301` (`parameters.vh`) |
| 규칙 | 정수 결과가 float 와 달라도 N·hop·체크포인트는 다시 고르지 않는다 (§10.9). 인접 정수 margin 만 본다 |

## A. Python 정수 기준 — `experiments/eval_streaming_int.py`

- 케이스 구성은 `eval_streaming` 과 **동일**(같은 seed, 같은 균형 선택, 같은 silence 접합).
  같은 인자면 float 결과가 그 도구를 재현한다.
- 창마다 `fixed_accuracy.fixed_logits` 로 pooled 합(정수)을 계산.
  - `keyword_idx = argmax(pooled[0:10])` (동률은 낮은 인덱스)
  - `keyword_margin = max(pooled[0:10]) − max(pooled[10:12])`
  - `rtl/kws_tail.v` argmax 블록과 같은 정의. POOL_BITS 오버플로를 검사한다.
- `IntVote` 는 `rtl/kws_vote.v` 를 레지스터 단위로 옮긴 것.
  `tests/test_streaming_int.py` 가 float 쪽 `ConsecutiveVote`(gate = 거부 → quiet)와
  랜덤 3,000 시퀀스에서 검출이 같음을 확인한다.
- 정책 상수는 `rtl/gen/<tag>/parameters.vh` 에서 읽는다.

### 칩 벡터 (`--export`)

| 파일 | 내용 |
|---|---|
| `stream_frames.hex` | 케이스 × 300 프레임, 16 비트 (채널 c = bit c) |
| `stream_expected.hex` | 케이스당 40 비트: `[39:24] crc16 \| [23:20] target \| [19:10] det1 {v, win[4:0], idx[3:0]} \| [9:0] det0` |
| `stream_cases.csv` | ROM 케이스 ↔ 평가 케이스, 검출, hit/quiet_false/wrong |
| `stream_trace.csv` | 창별 idx / margin / accept / streak / cooldown / armed (디버깅) |
| `stream_manifest.json` | 형식, 정책, 부분집합 요약 |

- crc16: CRC-16/CCITT-FALSE, 창마다 `{idx[3:0], margin[21:0]}` 26 비트 MSB first.
  결론(검출)이 같아도 중간 창 값 하나가 다르면 잡는다.
- 21 창, N=5, cooldown 10 에서 검출은 최대 2 회 → 슬롯 2 개.

## B. RTL 하네스 — `kws_stream_selftest` (작성 예정)

```
ROM(케이스 × 300 프레임) → 프레임 구동기 → cmp[15:0] → kws_stream_core → 채점기
                                                        (창 + 네트워크 + 투표, 배포 코드)
```

- 케이스마다 core 리셋 (Python 도 케이스마다 상태를 새로 시작).
- **프레임 경계 정렬**: capture 의 프레임 틱에 맞춰 구동 (가장 위험한 부분).
- 비교: 검출 2 슬롯(유효·창·클래스), 창 21 개 crc16, overrun = 0.
- VIO: total, match, hit, wrong, quiet_false, 첫 불일치 케이스.
- XSim 1–2 케이스로 창별 값을 `stream_trace.csv` 와 대조.

## C. 칩

- 실시간 파라미터(`FRAME_CYCLES = 500000`) 그대로. 240 × 3 s ≈ 12 분.
- 성공: 두 번 실행 모두 match 240/240, overrun 0. 칩의 hit/wrong/quiet_false 는
  `stream_manifest.json` 의 부분집합 요약과 같아야 한다.

## 위험

| 위험 | 증상 | 대책 |
|---|---|---|
| 프레임 경계 어긋남 | crc 대부분 불일치 | 틱 동기 구동, 시뮬에서 먼저 확인 |
| 투표 재무장 순서 | 결론만 가끔 다름, crc 일치 | IntVote = kws_vote, 등가 테스트 |
| margin 부호/폭 | 경계 근처 창만 다름 | POOL_BITS 검사, 경계 창 수 출력 |
| overrun | overrun_count > 0 | 칩 카운터 확인 |
| 첫 창 시점 | 창 번호 한 칸 밀림 | trace 창 번호와 대조 |
