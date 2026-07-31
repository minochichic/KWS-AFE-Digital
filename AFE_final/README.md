# AFE_final — 하드웨어 충실 최종 설계 + 격리 ML 재학습

회로도: [`artifacts/afe_final_system.svg`](artifacts/afe_final_system.svg)
(2차 GIC ×16 + 마이크 프리앰프 + 캘리브레이션 임계, 소자값·전력 포함)

## 격리 규칙
- 이 폴더에서 **계획·문서·결과**만 관리. ML 코드 변경은 최소로, 있으면 명시.
- `AFE/`, `AFE_highorder/`, `AFE_micamp/`, `AFE_tuning/` 및 기존 결과 **동결**.
- 복귀점: `git checkout phase-b-2ndorder` (2차 baseline), 이 폴더는 add-only.

---

## 0. 이 폴더가 생긴 이유 — Cerutti 정규화 재해석 (중요)

Cerutti IV-A 원문:
> "The full-precision envelopes' values are normalized using min-max scaling, and **the same
> min-max values are used to scale the initial thresholds**. The analog front end threshold can be
> adapted to this optimal threshold by **selecting the corresponding resistor divider**."

→ **논문의 min-max는 고정(데이터셋 레벨) 스케일링**이다. 근거: 같은 lo/hi로 threshold까지
스케일하고 그것을 **고정 저항 분압기**로 사상한다 — per-clip이면 분압기가 발화마다 바뀌어야
하므로 불가능. (논문이 "dataset-level"을 명시하진 않음: **강한 함의**, 확인 필요.)

**수학적 귀결**: 고정 lo/hi의 min-max는 아핀 변환이므로
`(x−lo)/(hi−lo) > thr  ⟺  x > lo + thr·(hi−lo)` → **고정 min-max ≡ 절대 임계(`normalize="none"`)**.

**따라서**:
- 우리 코드의 **per-clip min-max는 논문에서 이탈한 이상화**(=비인과 AGC, 하드웨어 대응물 없음).
- **AGC를 만들 필요가 없다.** `normalize="none"`이 오히려 **논문 충실 + 하드웨어 실현 가능**.
- 대가: 라우드니스 불변성 상실 → 그래서 **프리앰프(레벨 확보)가 더 중요**해진다.

---

## 1. 실험 설계

기준선(태그 `phase-b-2ndorder`): spice+√+per-clip minmax = **test 0.802**,
오프셋 주입 평가 시 vos=0.035에서 **0.47**, 오프셋-인지 학습 후 **0.73**.

### E1. 캘리브레이션 재평가 (재학습 불필요, 가장 저렴) ⬅ 먼저
기존 오프셋-인지 모델(`sc_v2_spice_sqrt_vos035`)을 **작은 vos로 평가**한다.
- 근거: 비교기 오프셋은 **소자별 고정 DC 오차**이므로 **1회 캘리브레이션으로 상쇄 가능**
  (동료 GUI의 `Vmargin → R7/R8` 기능이 바로 그 메커니즘).
- 즉 vos=0.035는 **미캘리브레이션 소자**를 뜻하고, 캘리브레이션 후 잔차는
  **트림 해상도 수준(vos ≈ 0.005–0.01)**.
- 평가만: `afe.cfg.comparator_vos = 0.005 / 0.01 / 0.02` 로 test acc 측정.
- **기대**: 0.73 → 0.78~0.80 근처. 남은 격차가 "진짜 오프셋 비용".

### E2. 절대 임계 학습 (논문 충실) — 핵심 실험
`normalize="none"` + spice + √ 로 재학습.
- 이러면 threshold가 **절대 스케일**이 되어 **고정 R7/R8로 직접 사상** 가능(도메인 왜곡 0).
- 비교: per-clip minmax 0.802 vs 절대 임계 ?
- **가설**: 라우드니스 불변성 상실로 소폭 하락 예상. 그 하락폭이 "AGC를 포기하는 비용"이고,
  그 대신 **하드웨어 실현 가능성**을 얻는다. (Cerutti가 지불한 것과 같은 비용.)

### E3. 절대 임계 + 고정 오프셋 (E2 위에)
E2 모델을 오프셋-인지로 학습 + 캘리브레이션 잔차로 평가.
- 여기서 **프리앰프 이득이 처음으로 ML 정확도에 영향**을 준다: 절대 스케일에서는
  신호는 G배 커지지만 오프셋은 그대로 → **실효 SNR_offset이 G배 개선**.
- 비교: 프리앰프 레벨 vs 무프리앰프 레벨 → **프리앰프의 정확도 가치를 숫자로 확정**.
  (per-clip minmax에서는 이득이 소거돼 측정 불가였던 것)

### E4. (선택) 노이즈 증강 강건성
증강 on/off × 깨끗한/노이즈(SNR 10±5 dB) testset 2×2. "깨끗한 정확도 비용 vs 강건성 이득".

---

## 2. 구현 방식 — 로컬 검증으로 확정 (중요)

`normalize="none"`을 그냥 쓰면 **STE가 망가진다**는 것을 로컬 GSC로 실측:

| 방식 | env 범위 | STE 창(`ste_clip`) 안 비율 | 코드 |
|---|---|---:|---|
| `none` + ste_clip=1.0(기본) | [0.001, **68.7**] | **54.6%** ⚠️ 45%가 그래디언트 0 | 0줄 |
| `none` + **ste_clip=4.0** | 동일 | 93.8% | 0줄 (config) |
| **고정 데이터셋 min-max** ⬅ 권장 | **[0, 1]** | **100%** | ~5줄 |

**고정 min-max를 권장하는 이유** (= Cerutti 방식):
1. **절대 임계와 이진 출력이 완전히 동일함을 증명**(로컬 검증: `(env≥thr) == (env_f≥thr_f)` 全True).
   아핀 변환이라 당연하며, 따라서 **고정 R7/R8로 직접 사상 가능**.
2. env가 [0,1]에 유지되어 **`ste_clip`·학습률 등 하이퍼파라미터를 건드릴 필요 없음**
   (`none`은 스케일이 68배라 STE 창을 다시 튜닝해야 함 = 교란 변수 추가).
3. 구현: 학습 시작 전 학습셋 일부로 lo/hi를 **한 번** 계산해 buffer로 고정
   (per-clip 계산을 상수로 교체). `normalize: "fixed"` 옵션 ~5줄.

| 실험 | 변경 | 규모 |
|---|---|---|
| E1 | **없음** (`comparator_vos`만 평가 시 세팅) | 0줄 |
| E2/E3 | `normalize: "fixed"` 추가 (lo/hi buffer) | ~5줄 |
| E2′/E3′ (대안, 코드 0) | `normalize='none'` + `ste_clip=4.0` | 0줄 |
| E4 | 없음 (`data.aug_noise_prob` 존재) | 0줄 |

## 3. 실행 (RTX)

```python
# E1 — 캘리브레이션 재평가 (재학습 없음)
ck = torch.load('runs/sc_v2_spice_sqrt_vos035/best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ck['model']); afe.load_state_dict(ck['afe'])
for vos in [0.0, 0.005, 0.01, 0.02, 0.035]:
    afe.cfg.comparator_vos = vos
    print(vos, np.mean([trainer.evaluate(test_loader)['acc'] for _ in range(3)]))

# E2' — 절대 임계 학습 (코드 변경 0 버전: ste_clip 조정 필수!)
cfg = load_config('configs/base.yaml', {'tag':'af_abs_thr',
    'afe.filterbank_source':'spice', 'afe.compression':'sqrt',
    'afe.normalize':'none', 'afe.ste_clip':4.0})     # ste_clip=1.0이면 45% 그래디언트 소실

# E3' — 절대 임계 + 오프셋 인지 (vos는 이제 절대 스케일 단위 주의)
cfg = load_config('configs/base.yaml', {'tag':'af_abs_vos',
    'afe.filterbank_source':'spice', 'afe.compression':'sqrt',
    'afe.normalize':'none', 'afe.ste_clip':4.0, 'afe.comparator_vos':0.05})
```
