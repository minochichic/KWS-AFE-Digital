# AFE 설정 레퍼런스 (`configs/base.yaml`의 `afe:`)

AFE(아날로그 프론트엔드 시뮬)는 파형을 1비트 시간-주파수 이미지 `[n_channels, T]`로
바꾼다. 구현: [`data/afe.py`](../data/afe.py) `AFEFrontend`, 스키마:
[`train/config.py`](../train/config.py) `AFEConfig`. 전체 흐름은
[`docs/diagrams/01_pipeline_overview.svg`](diagrams/01_pipeline_overview.svg) 참고.

```
[16000] 파형
  → STFT(stft_win_ms/stft_hop_ms, n_fft) → 스펙트로그램 ~100프레임
  → Mel n_channels필터(f_min–f_max, mel 등간격) → log
  → 엔벨로프(envelope_win_ms 창, envelope_reduce) → native_T
  → normalize → 채널별 threshold(ste, ste_clip) 비교 → {-1,+1}
[n_channels, T]
```

---

## ① 입력 정의

### `n_channels` = 16
AFE 출력 채널 수 = **Mel 필터 개수** = 모델 입력 `[16, T]`의 행 수. 하드웨어
필터뱅크 크기를 직접 결정하는 **핵심 sweep 축**. 늘리면 주파수 해상도↑ → 정확도↑
여지(Cerutti 8ch 76.3% → 64ch 86.0%)지만 하드웨어 비용↑. 헌장(CLAUDE.md 2.1)이
16으로 고정하고 있어 바꾸려면 상의 필요. `model.in_channels`와 반드시 같아야 함
(`validate()`가 강제).

### `sample_rate` = 16000
오디오 샘플링레이트(Hz). Speech Commands가 16 kHz. `f_max`는 이 값의 절반(Nyquist,
8000 Hz) 이하여야 함.

### `clip_ms` = 1000.0
클립 길이(ms). 1초 = 16000 샘플. 입력이 짧으면 zero-pad, 길면 crop
(`_fix_length`). `native_T` 계산의 분자.

---

## ② STFT (단시간 푸리에 변환)

### `stft_win_ms` = 25.0
STFT 윈도우 길이(ms). 25 ms = 400 샘플. 한 번에 주파수 분석하는 시간 구간. 길수록
주파수 분해능↑·시간 분해능↓.

### `stft_hop_ms` = 10.0
윈도우 이동 간격(ms). 10 ms = 160 샘플 → 1초에 약 100 STFT 프레임. **원시 시간
해상도**를 정한다(엔벨로프 단계에서 다시 요약됨).

### `n_fft` = 512
FFT 크기(주파수 bin 수 = n_fft/2+1 = 257). `win_length`(400) 이상이어야 함. Mel
필터가 이 주파수 축 위에서 만들어진다.

---

## ③ Mel 필터뱅크

### `n_mels` = 64  ⚠️ 현재 미사용
**AFE 경로에서 쓰이지 않는다.** `data/afe.py`의 `MelSpectrogram`은 `n_mels`를
`cfg.n_channels`(16)로 설정하므로, 실제 필터는 16개다. 이 `64`는 나중에 비교용
**full-precision Mel baseline**(Cerutti VI-A, 64-mel)을 구현할 때 쓰려고 남겨둔
자리이고 그때까지는 무시된다. `n_channels`(AFE 필터 수)와 혼동 금지.

### `f_min` = 50.0
필터뱅크 최저 코너 주파수(Hz).

### `f_max` = 8000.0
필터뱅크 최고 코너 주파수(Hz). **Nyquist(`sample_rate`/2 = 8000) 이하** 필수.
Cerutti Fig.6에서 50 Hz–8 kHz가 8-filter 최적 범위였다.

> **기록**: sc_v2 baseline 실행(test 0.809)은 `f_max=7500`으로 돌았다. Cerutti의
> 최적값(8 kHz)에 맞추고 이전 주석과의 불일치를 없애려 **8000으로 상향**했다. 즉
> 지금 base.yaml은 sc_v2와 이 한 값이 다르다 — 다음 실행은 이 차이를 감안할 것.

`f_min`~`f_max` 사이를 **Mel 척도 등간격**으로 16개 필터 중심을 잡는다(Hz로는
저주파 촘촘, Fig.3). "Mel 등간격 = Hz 등차수열"이 아님은
[`docs/first_task_result.md`](first_task_result.md)에 검증돼 있다.

---

## ④ 엔벨로프 — 출력 시간축 `T`를 결정

### `envelope_win_ms` = 10.0
**출력 시간 스텝 수를 결정**: `native_T = clip_ms / envelope_win_ms = 1000/10 = 100`
→ 128로 zero-pad(MatchboxNet 4.1). 이 창 안의 스펙트로그램을 한 시간 스텝으로 요약
(adaptive pooling). STFT hop(10 ms)과 독립적으로 **최종 T 해상도**를 정한다.

> ⚠️ **25 ms로 바꾸지 말 것**: native_T=40 → conv1 stride 2 후 20프레임 → conv2
> kernel span 57보다 짧아 conv2가 padding만 훑음 → 성능 붕괴. 자세히:
> [`docs/diagrams/08_dilation_span.svg`](diagrams/08_dilation_span.svg),
> `Config.time_axis_report()`. 유효 범위는 T≥128(10 ms 기준).

### `envelope_reduce` = max
창 안 요약 방식. `max` = 최대값(Cerutti IV-A "maximum values in windows").
`mean`(평균)도 선택 가능 — 미세 튜닝 후보.

---

## ⑤ 이진화 (comparator + 학습 가능 threshold)

### `normalize` = minmax
엔벨로프를 [0,1]로 정규화. **클립 전체(채널 통합) 기준**이지 채널별이 아니다 —
채널 간 레벨 차이를 보존해야 threshold가 그 차이를 흡수한다. 채널별로 하면 정보가
이중으로 죽는다(테스트로 잠금). `none`도 가능하나 비권장.

### `threshold_init` = channel_mean
16개 threshold 초기값 = **채널별 평균 엔벨로프**(Cerutti IV-A). 학습 시작 전
`init_thresholds(waves)`로 첫 배치에서 설정한다.

### `threshold_trainable` = true
threshold를 학습 파라미터로 둘지. `true`면 네트워크와 **end-to-end 동시 학습**
(CLAUDE.md 2.4). `false`면 초기값 고정(차선).

### `ste` = hardtanh
계단 함수(threshold 비교)의 **backward 근사**. `sign`은 미분이 0이라 그대로면
gradient가 끊긴다 → STE(straight-through estimator) 필수. `hardtanh`는 |x|<=clip
구간에서 gradient를 통과시킨다.

### `ste_clip` = 1.0
STE gradient가 흐르는 창 폭. `|env - thr| <= ste_clip`이면 gradient 통과. 엔벨로프가
[0,1], threshold도 그 근처라 항상 이 창 안 → 모든 threshold가 학습된다.

---

## 실험에서 만질 만한 것

| 변수 | 효과 | 비고 |
|---|---|---|
| `n_channels` | AFE 크기·정확도 | 하드웨어 직결, 헌장 고정(상의 필요) |
| `f_max` | 필터 커버 대역 | 8000이 Cerutti 최적. 이미 반영 |
| `f_min` | 저역 커버 | 50이 기본 |
| `envelope_win_ms` | 시간축 T | 10 ms 안전(conv2 제약), 바꾸면 broken 위험 |
| `envelope_reduce` | max/mean | 싼 튜닝 후보 |
| `normalize`, `ste_clip` | 이진화 민감도 | 미세 튜닝 |

**주의**: 한 번에 한 변수만 바꿔야 효과를 분리할 수 있다. `f_max`를 이미 8000으로
올렸으므로, 다음 실행의 delta에는 그 영향이 섞여 있다(sc_v2는 7500).
