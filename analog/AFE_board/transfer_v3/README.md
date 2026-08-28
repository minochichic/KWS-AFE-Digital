# 동료 아날로그 모델 전송본 v3 (2026-08-27)

`transfer_v3_20260827_201857_ba6ef035f9`, schema 3, `circuit_mode: preamp`,
토폴로지 `microphone_preamp_gic_active_detector`.

동료가 회로를 **전달함수로 만들어** Speech Commands 클립을 통과시킨 결과다.
우리 `data/afe.py` 가 소프트웨어로 흉내내던 체인 전체 — 필터 + 검출기 + 비교기 —
가 여기서는 이미 적용돼 있다.

| | |
|---|---|
| `spectrogram-v1.tar.gz` | **38,546 클립**, 각 `[16, 100]` {0,1} CSV |
| `model_package_snapshot/` | 채널별 회로 모델 (필터·검출기·동작점·검증) |

## 왜 tar 인가

낱개 CSV 로 커밋하면 **161 MB / 38,546 오브젝트**라 이후 모든 clone·status·checkout
이 영구히 느려진다. tar.gz 는 **5.8 MB 한 덩어리**이고, 순차 읽기가 낱개보다 오히려
빠르다(3초 vs 6초). `data/analog_spectrogram.py` 가 tar 를 직접 읽고, 디렉터리를
줘도 **비트 단위로 같은 결과**를 낸다(검증함).

첫 로드 후 `_cache_spectrogram-v1.npz`(1.9 MB)로 캐시한다. 이건 파생물이라
`.gitignore` 에 있다.

## 조건

| | |
|---|---|
| 마이크 입력 | 5 mVpp |
| 임계값 | ch0–3 0.98 V, ch4–11 0.99 V, ch12–15 0.97 V (5 mV 양자화) |
| 내부 샘플레이트 | 64 kHz (16 kHz × 4 오버샘플) |
| 비교기 | 모델에 미포함 (`comparator_included: false`) — 임계 비교는 외부에서 `v_env >= v_thr` |

**임계값은 동료가 임의로 정한 값이다** (본인 확인). 최적화된 값이 아니다.

## ⚠️ 10 클래스뿐이다

키워드 10개만 있고 `_silence_` 도 `_unknown_` 도 없다. 그래서 여기서 나온 정확도는
**`bd_base` 의 0.825(12클래스)와 직접 비교할 수 없다.** 12클래스에서 `_unknown_` 은
우리 최악(0.644), `_silence_` 는 최고(0.993)라 둘을 빼면 숫자가 어느 쪽으로 움직일지
산수로 정해지지 않는다.

비교 가능한 숫자를 얻으려면 동료에게 **`_background_noise_` 크롭(silence)과 나머지
25개 단어 일부(unknown)** 를 같은 체인으로 뽑아달라고 해야 한다.

당장은 **10클래스끼리** 비교하면 된다: `bd_base` 의 혼동행렬에서 키워드 10개만
재집계하면 같은 축이 되고, 그 차이가 **"우리 AFE 시뮬이 실제 회로와 얼마나 다른가"**
의 첫 정량적 답이다.

## 학습

```bash
python -m train.train --config configs/base.yaml --tag an_v3 \
  model.n_classes=10 model.stages.conv4.channels_abs=10 \
  data.analog_csv_root=analog/AFE_board/transfer_v3/spectrogram-v1.tar.gz
```

분할은 Speech Commands **공식 목록**을 쓴다(CSV 파일명이 원본 wav 이름과 같다).
그래서 `data.root` 가 여전히 필요하다 — 목록 파일을 거기서 읽는다.

## 데이터 상태 (전수)

죽은 채널 0개, 포화 채널 0개. 발화율 4~24%(평균 18.0%), **10.61 / 16 비트**
(우리 `bd_base` 는 11.16). 최하단 ch0 이 4% 로 다소 어둡다.
