# v15 검증 기록

검증일: 2026-07-31

## 기준 자료

- 최신 netlist: `netlist(6).txt`
- 최신 회로도: `sch(1).pdf`
- 2020 detector 비교 논문:
  `2020_Characterization_and_comparison_of_envelope_detectors_for_wake-up_sensor_interfaces_at_audio_frequencies(1).pdf`

회로와 netlist를 대조해 다음 canonical net을 확인했습니다.

```text
/vin
/v_filt
/v_env
/v_thr
/v_comp
```

V4의 terminal은 `/vin`과 `Net-_V3-Pad1_`입니다. OPA379, LPV7215,
BAT54WT1 model 이름과 회로 연결은 제공 자료를 그대로 사용했습니다.

## 자동 검사 결과

- 기존 v14 회귀검사: 59개 통과
- 새 AC와 Active Detector 회귀검사: 22개 통과
- 전체: 81개 통과
- Python compile 검사: 통과
- 16채널 AC/OP netlist 생성 검사: 통과
- 16채널 Detector netlist 생성 검사: 통과
- fake ngspice Detector raw parser와 CSV 검사: 통과
- 최신 회로도 image 크기와 overlay node 검사: 통과
- production과 문서의 이전 voltage net 이름 잔존 검사: 통과
- 새 함수와 class docstring 검사: 통과
- 기존 공개 import와 `SweeperGUI` controller method 검사: 통과

기본 `channel_components.csv`의 SHA-256은 수정 전후 모두 다음 값입니다.

```text
338b282110eebb5f15de008aa1a4a977b0721a523cfe1c9456ca086f5bfe2fe3
```

## 실제 model simulation 상태

실제 OPA379, LPV7215, BAT54WT1 model을 사용하는 Windows ngspice 해석은
실행하지 않았습니다. 사용자 확인 전 제한에 따라 netlist 생성, parser,
합성 waveform, fake ngspice 검사까지만 수행했습니다.

## 공통 인프라 개선 검증 (2026-08-01)

parser 성능, 결과 CSV 선택 생성, 실패 진단 통일을 적용하고 검증했습니다.

### parser 동등성

기존 parser 구현을 그대로 복제해 신·구 출력을 비교했습니다.

- `results/` 안의 실제 rawfile 62개 전수 비교: 불일치 0건
- 비교 대상에는 5변수 transient/detector raw와 530변수 op raw가 포함됩니다.
- 중단된 실행으로 잘린 rawfile은 신·구 parser가 모두 동일하게 거부합니다.
- `read_transient_raw()`와 `read_detector_raw()`의 열 선택 결과도
  기존 전체 parse 후 열 선택 결과와 완전히 일치합니다.

### parser 성능

`tran.raw` 80.6 MB, 635,355점, 5변수 기준이며 page cache를 채운 뒤
3회 중 최소값입니다.

| 항목 | 기존 | 개선 후 |
| --- | --- | --- |
| `read_ascii_rawfile()` | 4.35 s | 1.08 s |
| transient 읽기 end to end | 4.84 s | 1.15 s |
| 피크 메모리 | 420 MB | 143 MB |

시뮬레이션 결과 데이터는 바꾸지 않았습니다. `.option INTERP`와 binary
rawfile 전환은 결과가 달라지므로 적용하지 않았습니다.

### 결과 CSV 선택 생성

fake ngspice test double로 두 runner를 실제 실행해 확인했습니다.

- `write_csv=False`: `transient.csv`와 `detector.csv` 미생성, rawfile과
  `detector_settings.csv`는 생성, rows 수 동일, Rise time 측정 동일
- `write_csv=True`: 기존과 동일하게 생성
- GUI 기본값은 두 탭 모두 해제, 체크 시 설정 객체에 정상 반영

### 실패 진단 통일

실패하는 ngspice stand in으로 AC/OP, Transient, Detector 세 경로를 모두
실패시켜 확인했습니다.

- 로그 앞부분의 오류 행 3개를 행번호와 함께 표시
- 기존 tail 40행 방식은 같은 로그에서 해당 오류를 전혀 포함하지 못함
- 세 경로 모두 PSpice 호환 힌트, 모델 include 힌트, 두 로그 경로 포함
- Detector 경로에 없던 로그 파일 경로가 추가됨

### 자동 검사 결과

- 전체 회귀검사: 108개 통과 (기존 81개 + 신규 27개)
- Python compile 검사: 통과
- GUI 구성 검사: 16채널 로드, 새 체크박스 기본 해제 확인

실제 model을 사용하는 ngspice 실행 결과는 아래
`실제 model ngspice 실행 검증` 절에 있습니다.

## 실제 model ngspice 실행 검증 (2026-08-01)

사용자 확인을 받아 실제 OPA379, LPV7215, BAT54WT1 model로 처음 실행했습니다.
ngspice는 `C:\Spice64\bin\ngspice_con.exe` (ngspice-46)입니다.

### Transient 실행 실패 원인과 해결

증상은 `ngspice 종료 코드 1`과 launcher 로그의
`...\ch00\tran_ngspice.log: No such file or directory`였습니다.

원인은 Windows MAX_PATH 초과입니다. 프로젝트 root가 167자이고 transient가
`stim_0001_yes_004ae714_nohash_0` 같은 긴 폴더를 추가하면서 생성 경로가
261에서 264자가 되었습니다. Python은 이 경로에 파일을 만들 수 있지만
`ngspice_con.exe`는 열지 못합니다. Detector는 237자여서 정상 동작했고,
그래서 Transient에서만 증상이 나타났습니다.

| 경로 | 변경 전 | 변경 후 |
| --- | --- | --- |
| transient `channel_XX_params.inc` | 264자 (초과) | 248자 |
| transient `channel_XX_tran.cir` | 261자 (초과) | 246자 |

stimulus 폴더 이름을 `stim_0001_단어`로 고정하고, 실행 직전 경로 길이 검사를
추가했습니다.

### 실행 결과

| 해석 | 결과 |
| --- | --- |
| AC + OP, 3채널 | 0.7 s, 정상 |
| Active Detector, 200 ms preset | 24.0 s, 294,146점 |
| Transient, 3단어 × 1채널, 0.2 s | 9.0 s, 3 jobs 정상 |

AC 측정값은 설계값과 다음과 같이 일치합니다.

| 채널 | 설계 fc | 측정 f0 | 오차 | gain | Q |
| --- | --- | --- | --- | --- | --- |
| ch00 | 166 Hz | 167.7 Hz | 1.0 % | 6.03 dB | 1.34 |
| ch08 | 2042 Hz | 2061.1 Hz | 0.9 % | 6.65 dB | 5.33 |
| ch15 | 6761 Hz | 6803.5 Hz | 0.6 % | 8.40 dB | 7.08 |

Detector는 Headroom 9.020 mV, Rise 0.135 ms, Fall 1.469 ms를 측정했습니다.
`write_csv=False`에서 `detector.csv`와 `transient.csv`가 생성되지 않고 rawfile,
그래프 데이터, 측정 결과는 정상임을 확인했습니다.

### 확인된 회로 관찰 사항

세 채널 모두 DC `Vmargin = Vthr - Venv_DC`가 음수입니다.

| 채널 | Venv_DC | Vthr | Vmargin |
| --- | --- | --- | --- |
| ch00 | 917.83 mV | 917.46 mV | -0.37 mV |
| ch08 | 917.83 mV | 912.24 mV | -5.59 mV |
| ch15 | 917.83 mV | 913.50 mV | -4.33 mV |

기본 `channel_components.csv`의 R7, R8 값 기준이며 code 문제가 아닙니다.
비교기가 정지 상태에서 이미 넘어가 있다는 뜻이므로, DC 탭의 목표 Vmargin
기능으로 R7과 R8을 다시 잡아야 합니다.

### 데이터셋 취급

`speech_commands_pwl`은 읽기만 했고 변경하지 않았습니다. 실행 전후 최상위
항목은 동일하게 12개입니다.

## Transient 실행 중 결과 확인 검증 (2026-08-01)

프로젝트를 `10_AI_Project\ngspice_v15_2608001` (89자)로 옮긴 뒤 검증했습니다.
최장 생성 경로는 171자이며 Windows 한계 대비 88자 여유가 있습니다.

실제 ngspice로 PWL 1개 × 3채널, stop time 0.3 s를 실행하며 GUI 상태를
0.05초 간격으로 관측했습니다.

| 경과 | 완료 job | 그래프 | 선택 가능 채널 |
| --- | --- | --- | --- |
| 0.0 s | 0 | 없음 | 없음 |
| 5.3 s | 1 | 있음 | ch00 |
| 9.1 s | 2 | 있음 | ch00, ch01 |
| 13.1 s | 3 | 있음 | ch00, ch01, ch02 |

변경 전에는 13.1초가 지나 전체 job이 끝날 때까지 그래프가 나오지 않았습니다.

### 실행 중 표시 보존 검증

ch00이 표시된 상태에서 커서를 하나 배치하고 남은 두 job과 실행 종료 처리를
모두 통과시킨 뒤 확인했습니다.

- 표시 result 객체 동일: 유지
- 그래프 위젯 재생성 안 됨: 유지
- 배치한 커서 1개: 그대로 유지
- 표시 채널 ch00: 유지
- 선택 가능 채널: ch00, ch01, ch02

즉 나중에 끝난 job이나 실행 종료가 사용자가 보고 있던 그래프를 다시 그리지
않습니다.

### 자동 검사 결과

- 전체 회귀검사: 114개 통과 (직전 108개 + 신규 6개)
- 새 위치에서 AC/OP, Detector, Transient 실제 model 실행 모두 정상이며
  측정값은 이동 전과 동일합니다.

## U3 opamp 선택 검증 (2026-08-01)

### 핀 순서 대조

`10_AI_Project\lib`의 subckt 선언을 직접 읽어 대조했습니다.

| 모델 | subckt 선언 | 핀 순서 |
| --- | --- | --- |
| OPA379 | `.SUBCKT OPA379 1 3 5 2 4` | +IN, −IN, +V, −V, OUT |
| TLV9041D | `.subckt TLV9041D IN+ IN- VCC VEE OUT` | 동일 |
| TLV9042 | `.subckt TLV9042 IN+ IN- VCC VEE OUT` | 동일 |

현재 netlist의
`XU3 Net-_U3-+_ Net-_D1-A_ 1.8V GND Net-_D1-K_ OPA379`
와 일치하므로 노드 재배열 없이 모델 이름만 교체합니다. `lib\BAT54WT.lib`의
모델명도 `Dbat54wt1`로 기존과 같습니다.

### 실제 model 실행 결과

ch00, 1.8 V 단전원, gate 50~250 ms, 10 mVpp @ 1 kHz 조건입니다.

| opamp | 상태 | 점수 | 시간 | v_env 최소 | v_env 최대 | swing |
| --- | --- | --- | --- | --- | --- | --- |
| OPA379 | 정상 | 294,146 | 21.0 s | 913.50 mV | 929.29 mV | 15.79 mV |
| TLV9041D | 정상 | 249,086 | 43.9 s | 937.79 mV | 952.85 mV | 15.05 mV |
| TLV9042 | 정상 | 230,011 | 18.1 s | 942.05 mV | 962.64 mV | 20.59 mV |

세 모델 모두 수렴했고 생성된 netlist에서 XU3 노드 순서가 보존됨을 매 실행
확인했습니다. 필터부 XU1, XU2는 OPA379로 유지됩니다.

### 확인된 설계 영향

모델을 바꾸면 `v_env`의 DC 기준점이 크게 이동합니다.

- OPA379 대비 TLV9041D는 약 +24 mV, TLV9042는 약 +29 mV
- ch00의 `Vthr`는 917.46 mV이므로, TLV 모델에서는 `Venv_DC`가 `Vthr`를 훨씬
  크게 웃돌아 `Vmargin`이 더 깊은 음수가 됩니다

즉 **opamp를 바꾸면 R7과 R8을 다시 잡아야 합니다.** DC 탭의 목표 Vmargin
기능으로 재계산할 수 있습니다. TLV9042는 swing이 20.59 mV로 OPA379의
15.79 mV보다 큽니다.

## GUI 개편 검증 (2026-08-01)

- Detector 탭: preset, 안내문, 채널 드롭다운, Headroom/Rise/Fall 패널과
  그래프의 Vlow/Vhigh 선·교차 marker를 제거했습니다. 채널은 다른 탭과 같은
  목록창을 씁니다. 파일은 1126행에서 931행이 되었습니다.
- `measure_detector_response()`, `default_detector_levels()`,
  `DetectorSettings.quick_200ms_preset()` 등은 공개 API 호환을 위해 함수로
  남기고 GUI 연결만 끊었습니다.
- Detector 다중 채널 순차 실행과 `detector_result` 이벤트를 추가했습니다.
  사용자가 중지하면 예외 대신 정상 종료로 처리되어 오류 대화상자가 뜨지
  않습니다.
- 네 탭 채널 목록에 `소자값 보기` 버튼을 추가했습니다. 표 내용은 기존
  `_component_rows()`를 재사용합니다.

### 회로도 overlay 수정

`circuit_template.png`(1280×590)에서 `{변수}` 자리표시자 13개의 실제 픽셀
박스를 연결요소 분석으로 측정해 좌표를 맞추고, 글꼴을 11 pt에서 8 pt로,
여백을 2 px에서 1 px로 줄이고 표시 문자열을 `10.61 kΩ`에서 `10.61k`로
축약했습니다.

자리표시자가 아닌 기존 그림을 덮는 픽셀 수를 실측했습니다.

| | 가린 잉크 | 최악 소자 |
| --- | --- | --- |
| 변경 전 (11 pt, `kΩ`, 여백 2) | 549 px | R8 114 px |
| 변경 후 (8 pt, 축약, 여백 1) | 9 px | R8 9 px |

남은 9 px는 R8 저항 외곽선의 1픽셀 열입니다. U3 옆 `OPA379` 텍스트 위치
`(807,228)`에는 선택된 모델명을 그리며, 기본값이 아니면 다른 색으로 표시합니다.

### 전체 검사

- 회귀검사 126개 통과 (직전 114개 + 신규 12개)
- GUI 구성 확인: Detector 채널목록 16개, opamp 콤보 3개 모델,
  삭제된 preset/Vlow/Vhigh 변수 부재 확인

## 모델 라이브러리 경로 통일 (2026-08-01)

`netlist_template.cir`의 고정 모델 include 세 줄을 `KiCAD\myLib`에서
`10_AI_Project\lib`으로 옮겼습니다. 선택형 U3 opamp가 쓰는
`MODEL_LIBRARY_DIR`와 같은 폴더가 되어 버전 불일치 가능성이 사라졌습니다.

교체 전후 파일이 동일함을 MD5로 확인했습니다.

| 기존 | 신규 | MD5 |
| --- | --- | --- |
| `myLib\BAT54WT1.REV2.LIB` | `lib\BAT54WT.lib` | `469f97de13c0…` |
| `myLib\lpv7215\lpv7215.lib` | `lib\LPV7215.lib` | `416aa7ce2403…` |
| `myLib\op379\…\OPA379.LIB` | `lib\OPA379.LIB` | `dde8b1e64fa3…` |

이름만 다른 같은 파일이므로 결과가 바뀌지 않아야 하며, 실제 실행으로
확인했습니다.

| 항목 | 통일 전 | 통일 후 |
| --- | --- | --- |
| ch00 f0 / gain / Q | 167.7 Hz / 6.03 dB / 1.34 | 동일 |
| ch15 f0 / gain / Q | 6803.5 Hz / 8.40 dB / 7.08 | 동일 |
| ch00 Venv_DC | 917.83 mV | 동일 |
| Detector 점수 / v_env | 294,146 / 913.498~929.290 mV | 동일 |

## opamp 전 해석 반영 수정 (2026-08-01)

DC와 Transient 탭에 opamp 선택기를 추가했지만 `make_detector_netlist()`만
모델을 교체하고 있었습니다. AC, OP, Transient netlist는 계속 OPA379를 써서
"모델을 바꿔도 결과가 같다"는 증상이 나왔습니다.

`apply_detector_opamp()` 공통 헬퍼를 만들어 네 생성기가 모두 XU3 모델명과
라이브러리 include를 처리하게 했습니다. XU3가 없는 축약 템플릿은 기본 모델일
때만 그대로 통과하고, 다른 모델을 지정하면 명확히 거부합니다.

### 실측 확인 (ch00, 실제 ngspice)

| U3 opamp | netlist XU3 | AC f0 | AC gain | Venv_DC |
| --- | --- | --- | --- | --- |
| OPA379 | OPA379 | 167.7 Hz | 6.03 dB | 917.832 mV |
| TLV9041D | TLV9041D | 167.7 Hz | 6.03 dB | 943.449 mV |
| TLV9042 | TLV9042 | 167.7 Hz | 6.03 dB | 948.243 mV |

AC의 f0와 gain이 같은 것은 정상입니다. 필터부 U1, U2는 OPA379 고정이고 AC
출력 node `/v_filt`는 검출기 앞단이기 때문입니다. `Venv_DC`는 모델마다
분명히 다릅니다.

## 소자값 보기 창이 비어 보이던 문제 (2026-08-01)

Treeview의 부모가 Toplevel이었는데 `in_`으로 하위 Frame에 grid해서 위젯이
매핑되지 않았습니다. Treeview를 그 Frame의 자식으로 생성하도록 고쳤습니다.
16채널 선택 시 17열 21행이 표시되고 가로/세로 스크롤이 모두 동작합니다.

## 공유 소자값 패널의 편집 채널 표시 (2026-08-01)

"DC나 Transient에서 공유 소자값을 바꿔도 Detector 패널이 안 바뀐다"는 보고를
재현했습니다. 동기화 코드 자체는 정상이며, 증상은 **탭마다 다른 채널을 보고
있을 때**만 나타났습니다.

```
Transient 목록에서 ch05 선택 후 R4=33 편집
  → AppState ch05 R4 = 33.0   (반영됨)
  → Detector 패널 R4 = 10.00  (ch00을 보는 중이라 그대로)
```

`_sync_shared_component_editor()`는 패널이 보는 채널과 편집된 채널이 다르면
건너뜁니다. 다른 채널 값을 덮어쓰지 않기 위한 의도된 동작이지만, 어느 채널을
편집 중인지 표시가 없어 동기화 실패로 읽혔습니다.

### 적용한 변경

- 패널 제목에 편집 채널 표시: `공유 소자값 · ch05`
- 채널 불일치로 건너뛸 때 상태줄에 이유 표시:
  `ch05가 다른 탭에서 변경됨 (이 패널은 ch00)`
- `_on_detector_channel_changed()`의 조기 반환 조건을 좁혀 패널 채널이
  목록 선택과 어긋난 상태에서 회복되도록 보강

### 발견해서 함께 고친 이름 충돌

공유 패널이 상태 변수를 `{prefix}_component_state_var`로 만들면서 Transient
탭이 이미 쓰던 탭 전체 상태줄 변수와 충돌했습니다. 패널 상태줄이 탭 상태줄
문구로 덮이고 있었습니다. 패널 전용 이름 `{prefix}_panel_state_var`로 분리해
두 라벨이 각자 동작하도록 했습니다.

### 실동작 확인

| 상황 | Detector 패널 | Transient 패널 |
| --- | --- | --- |
| 초기 | `· ch00` R4=10.00 | `· ch00` R4=10.00 |
| Transient ch05에서 R4=33 | `· ch00` R4=10.00, "ch05가 다른 탭에서 변경됨 (이 패널은 ch00)" | `· ch05` R4=33.00 |
| Detector도 ch05로 맞춘 뒤 R4=44 | `· ch05` R4=44.00, "다른 탭에서 변경됨" | `· ch05` R4=44.00 |

회귀검사 133개 통과 (직전 129개 + 신규 4개).

## 선택 채널 일괄 적용 (2026-08-01)

공유 소자값 패널에 `선택 채널 일괄 적용 (Vthr 제외)` 옵션을 추가했습니다.
기본 CSV에서 R4, R5, R6, C3, U3 opamp는 16채널이 모두 같은 값이므로 한 번에
적용하는 편이 자연스럽습니다. Vthr는 채널별 R7/R8 튜닝이라 패널 채널에만
적용해 다른 채널의 분압기를 덮어쓰지 않습니다.

### 실동작 확인 (ch00~ch05 상태)

| 단계 | ch00 | ch01 | ch02 | ch03 | ch04 | revision |
| --- | --- | --- | --- | --- | --- | --- |
| 초기 | R4=10 | 10 | 10 | 10 | 10 | 1 |
| 일괄 OFF, ch02에 R4=20 | 10 | 10 | **20** | 10 | 10 | 2 |
| 일괄 ON, ch00~03에 R4=55/TLV9042 | **55** | **55** | **55** | **55** | 10 | 3 |

- 선택하지 않은 ch04는 그대로입니다.
- 네 채널을 바꿨지만 revision 증가량은 1입니다.
- Vthr는 917.41 / 921.24 / 924.70 / 918.00 mV로 채널별 값이 유지됩니다.
- 패널 상태줄: `4개 채널에 일괄 적용됨 ch00, ch01, ch02, ch03`

회귀검사 138개 통과 (직전 133개 + 신규 5개).

## 채널 클릭이 소자 패널을 따라가지 않던 문제 (2026-08-01)

Detector 채널 목록에서 채널을 클릭해도 공유 소자값 패널이 `· ch00`에 머물렀습니다.

원인은 공용 `_build_channel_selector()`의 리스트박스 바인딩이
`_update_run_button()`만 호출하고 채널 변경 훅이 없던 것입니다. Transient는
자체 목록에 직접 바인딩해서 동작했고, 공용 빌더를 쓰는 Detector와 DC만
빠져 있었습니다.

- `_build_channel_selector()`에 `on_select` 인자를 추가
- Detector는 `_on_detector_channel_changed`를 연결
- DC는 `_on_dc_channel_selection_changed`를 추가해 목록 선택을 DC 결과 채널
  콤보로 전달 (결과가 없는 채널은 무시)

### 실동작 확인 (창을 띄우고 실제 선택 이벤트 발생)

각 탭을 raise한 뒤 목록을 클릭해 패널 값이 AppState와 일치하는지 확인했습니다.

| 탭 | 클릭 | 패널 제목 | Vthr | AppState |
| --- | --- | --- | --- | --- |
| Detector | ch15 | `· ch15` | 913.50 | 913.50 |
| Detector | ch03 | `· ch03` | 918.00 | 918.00 |
| Detector | ch09 | `· ch09` | 910.60 | 910.60 |
| Transient | ch15 / ch03 / ch09 / ch00 | 각각 일치 | 일치 | 일치 |
| DC | ch08 → 편집채널 ch08, R1=57.10 | | | 일치 |
| DC | ch15 → 편집채널 ch15, R1=53.40 | | | 일치 |

검증 중 노트북의 **보이지 않는 페이지에서는 Tk 가상 이벤트가 전달되지 않는다**는
점을 확인했습니다. 탭을 raise하지 않으면 클릭 이벤트가 핸들러에 도달하지
않으므로, 이후 GUI 이벤트 검증은 해당 탭을 먼저 선택해야 합니다.

회귀검사 142개 통과 (직전 138개 + 신규 4개).

## 2020 논문 preset 확인

논문 본문에는 sinusoidal signal duration이 1.5 s로 기재되어 있습니다.
따라서 GUI에는 요청된 `200 ms 빠른 preset`과 원문을 반영한
`2020 논문 1.5 s preset`을 구분해 제공합니다.

