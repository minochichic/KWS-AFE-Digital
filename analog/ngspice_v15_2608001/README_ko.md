# ngspice 16채널 AC, OP, Active Detector, Transient GUI

이 프로그램은 `channel_components.csv`의 16채널 소자값을 공통 회로에
적용해 다음 작업을 수행하는 Windows용 GUI입니다.

1. DC 동작점
2. AC 시뮬
3. Active Detector
4. Transient 시뮬
5. WAV to PWL 시뮬

기존 공개 import와 `SweeperGUI` 공개 메서드를 유지합니다. 기본
`channel_components.csv`는 읽기 전용이며, 편집값은 메모리와 새 버전 CSV,
각 실행의 `applied_components.csv`에만 기록합니다.

세부 데이터 흐름은 [ARCHITECTURE_ko.md](ARCHITECTURE_ko.md)에 있습니다.

## 1. 최신 회로 기준

회로 연결과 모델은 사용자가 제공한 `netlist(6).txt`와 `sch(1).pdf`를 서로
대조해 반영했습니다. 확인된 전압 net 이름은 다음과 같습니다.

- `/vin`
- `/v_filt`
- `/v_env`
- `/v_thr`
- `/v_comp`

`netlist_template.cir`은 다음 연결을 유지합니다.

- V4는 `/vin`과 `Net-_V3-Pad1_` 사이에 연결됩니다.
- OPA379을 사용하는 필터와 active detector 회로를 유지합니다.
- BAT54WT1 diode 두 개를 유지합니다.
- LPV7215 비교기는 `/v_env`와 `/v_thr`를 입력으로 받고 `/v_comp`를
  출력합니다.
- 전원과 0.9 V bias 연결을 유지합니다.

`circuit_template.png`는 최신 회로도 PDF에서 다시 만들었습니다. DC 탭의 소자값
overlay 좌표는 이미지에서 `{변수}` 자리표시자 라벨의 실제 픽셀 위치를 측정해
맞췄으므로, 적용값이 자기 자리표시자 위에만 그려지고 소자번호를 가리지 않습니다.
좌표는 `sweeper/constants.py`의 `SCHEMATIC_COMPONENT_POSITIONS`에 있습니다.

## 2. 준비와 실행

1. Windows ngspice를 설치합니다. 자동 실행에는 가능하면
   `ngspice_con.exe`를 사용합니다.
2. 모델 라이브러리 폴더를 실제 PC 경로에 맞춥니다. 기본값은
   `10_AI_Project\lib`이며 두 곳을 함께 바꿔야 합니다.

   - `netlist_template.cir`의 `.include` 세 줄 (BAT54WT, LPV7215, OPA379)
   - `sweeper/constants.py`의 `MODEL_LIBRARY_DIR` (선택형 U3 opamp용)

   고정 모델과 선택형 opamp가 같은 폴더를 쓰므로 버전이 어긋날 수 없습니다.
3. 패키지를 설치하고 GUI를 실행합니다.

```powershell
py -m pip install -r requirements.txt
py ngspice_channel_sweeper.py
```

또는 `run_gui.bat`을 실행합니다. 초기 상위 탭은 `DC 동작점`입니다. 실행
완료나 오류가 현재 상위 탭 또는 현재 결과 탭을 강제로 바꾸지 않습니다.

## 3. AC Magnitude의 dB와 Linear 표시

AC Magnitude의 X축은 기존처럼 `Linear`와 `Decade`를 선택할 수 있습니다.
Magnitude 단위는 `dB`와 `Linear` 중에서 선택합니다. 두 단위 모두 일반
linear Y축을 사용하며, Linear 단위에서 logarithmic 또는 decade Y축을
사용하지 않습니다.

변환식은 다음과 같습니다.

```text
Glinear = 10 to the power of (GdB divided by 20)
GdB = 20 times log base 10 of Glinear
```

- -20 dB는 0.1입니다.
- -40 dB는 0.01이며, -20 dB 아래의 값도 삭제하거나 0.1로 고정하지
  않습니다.
- 내부 데이터, 그래프, 커서, 중심주파수 Gain, 복사 결과는 정확한 변환값과
  현재 단위를 사용합니다.
- Linear 자동 Y범위는 0부터 실제 최대값에 5 percent 여백을 더한 값까지로
  설정합니다.
- dB와 Linear 수동 Y범위는 각각 저장되며 단위 전환, 채널 변경, 재실행 뒤에도
  유지됩니다.
- Q와 대역폭은 표시 단위와 무관하게 원본 dB 데이터로 계산합니다.
- -3.0103 dB의 Linear magnitude는 peak magnitude를 square root of 2로
  나눈 값입니다.

AC 출력의 기본 node는 `/v_filt`입니다. `ac.csv`에는 dB와 Linear magnitude,
phase를 함께 기록하며, `ac_metrics_summary.csv`에도 dB와 Linear Gain을 함께
기록합니다.

## 3.1 채널 목록의 소자값 보기

DC 동작점, AC 시뮬, Active Detector, Transient 네 탭의 채널 목록에는
`소자값 보기` 버튼이 있습니다. 채널을 하나 이상 고르고 누르면 선택한 채널의
소자 종류와 값을 나란히 비교하는 창이 열립니다.

표시 내용은 R1, RA1/RA2, R2/R3, R4~R8, C1/C2, C3와 고정 전원, 소자 모델,
그리고 채널별 U3 opamp, 설계 fc, 설계 Q, Vthr입니다. 16채널을 모두 선택하면
화면을 넘어가므로 가로와 세로 스크롤을 함께 제공합니다. 현재 AppState 값이므로
**다음 실행에 실제로 적용될 값**과 항상 일치하며, 창 상단에 현재
`component_revision`을 함께 보여줍니다.

## 4. DC 동작점과 공통 소자값

DC는 `.save all`과 `.op`를 사용합니다. 결과는 최신 회로도 위 node overlay와
전체 node 전압 목록으로 표시됩니다. 전압은 화면에서 mV, 결과 CSV에서 V로
보존합니다.

DC 동작점 탭에 보이는 mV 값은 모두 소수점 둘째자리까지 표시합니다. 회로도
overlay, node 전압 표, 계산 Vthr, 목표 V_margin, 최근 실제 Vmargin이 모두
같은 자릿수를 씁니다. 표시 자릿수일 뿐이며 rawfile과 CSV는 전체 정밀도를
그대로 보존하고, 계산도 반올림하지 않은 값으로 수행합니다. 자릿수는
`sweeper/constants.py`의 `VOLTAGE_DISPLAY_DECIMALS`로 바꿀 수 있습니다.

DC 동작점의 비교기 여유 전압은 다음 정의를 사용합니다.

```text
Vmargin = Vthr - Venv_DC
```

`Vthr`와 `Vmargin`은 같은 분압기를 보는 두 가지 방식이라 **둘 중 하나를
입력하면 나머지가 자동으로 정해집니다.** DC 탭에서 어느 칸을 편집하든 R7과
R8이 다시 계산됩니다.

DC 탭의 소자 편집 패널에서 U3 opamp도 고를 수 있습니다. Active Detector,
Transient 탭의 `공유 소자값`과 같은 AppState를 쓰므로 한쪽에서 바꾸면
회로도 그림, `고정 소자 / 모델` 표, 다른 탭 입력칸이 즉시 함께 갱신됩니다.

R7은 1.8 V 쪽, R8은 GND 쪽 divider 저항입니다. 목표 Vmargin을 적용할 때
R7과 R8은 0.01 kohm 단위로 독립 반올림하며, 합계를 특정 값으로 강제하지
않습니다. 소자 변경으로 `Venv_DC`가 바뀔 수 있으므로 첫 새 OP 결과를 기준으로
한 번만 보정한 뒤 최종 OP를 표시합니다.

모든 저항은 0.01 kohm 정밀도로 계산하고 표시합니다. R4 또는 R5를 바꾸면
R6를 다음 식으로 제안하지만, 사용자는 R6를 다시 독립적으로 수정할 수
있습니다.

```text
R6 = (R4 * R5) / (R4 + R5)
```

## 5. Active Detector

Active Detector는 WAV 또는 PWL 기반 Transient와 별도의 실행 경로입니다.
`DetectorSettings`, `DetectorResult`, detector netlist 생성기,
`DetectorSimulator`, 결과 cache를 따로 사용하며 GUI controller는
`sweeper/gui/detector_tab.py`에 있습니다.

Detector 실행은 Transient의 PWL 선택, 결과 cache, 축 범위, 커서, 선택
채널을 바꾸지 않습니다. 두 workflow가 공유하는 simulation 입력은 현재
AppState의 소자값과 `component_revision`뿐입니다. 소자 변경 전 Detector
결과에는 `소자 변경 전 결과, 재실행 필요`가 표시됩니다.

### 5.1 Gated sine 설정

채널은 다른 탭과 같은 목록창에서 고릅니다. 여러 채널을 선택하면 순차 실행하며,
Transient와 같이 **끝난 채널부터 바로 결과를 볼 수 있습니다**. 첫 결과만 자동으로
그리고, 이후에는 목록에서 채널을 골라야 표시가 바뀌므로 실행 중 보고 있던
그래프가 가로채이지 않습니다.

사용자는 다음 값을 편집할 수 있습니다.

- 정현파 주파수, Hz
- 입력 진폭, mVpp 또는 Vpp
- Gate ON 시각, ms
- Gate 지속시간, ms
- 전체 해석시간, ms
- maximum timestep, us

Gated sine은 기존 V4의 양단을 유지한 채 `/vin`에 인가합니다. `/v_filt`에
직접 인가하지 않으므로 필터를 우회하지 않습니다. Vpp는 실제 `/vin`에
인가되는 peak to peak 전압입니다. 실제 positive peak와 negative peak를 모두
포함하도록 Gate 지속시간은 정현파 한 주기 이상이어야 합니다.

maximum timestep이 한 sine 주기당 20개 sample보다 거칠면 실행 전에 경고를
표시합니다. 20 points per cycle은 phase 간격을 18 degree 이하로 두고 peak
누락 가능성을 낮추기 위한 최소 해상도 기준이며, 경고이므로 사용자가 확인하면
실행할 수 있습니다.

### 5.2 그래프 모드

하나의 큰 그래프에 다음 네 파형을 함께 표시합니다.

- `v_in`은 `/vin`
- `v_filt`는 `/v_filt`
- `v_env`는 `/v_env`
- `v_thr`는 `/v_thr`

기본 절대전압 모드는 시간 ms와 전압 mV를 사용합니다. AC 비교 모드는 Gate
ON 이전 평균 DC 값을 다음과 같이 제거합니다.

```text
vin_ac = vin - vin_pre
vfilt_ac = vfilt - vfilt_pre
venv_ac = venv - venv_pre
vthr_relative = vthr - venv_pre
```

AC 비교의 threshold 범례는 `v_thr relative`입니다. 진폭 정규화, 시간축
이동, 위상 정렬을 하지 않으므로 실제 필터 Gain과 phase 차이가 유지됩니다.
절대전압 모드와 AC 비교 모드의 수동 X/Y 범위는 각각 저장됩니다. Gate ON과
Gate OFF를 세로선으로 표시하며 기존 선 추종 다중 커서를 재사용합니다. 그래프
제목에는 채널과 함께 그 실행에 쓰인 U3 opamp 모델이 표시됩니다.

### 5.3 U3 opamp 선택

`공유 소자값`에서 active detector의 U3 opamp를 고를 수 있습니다. 같은 패널이
Transient 탭에도 있고 DC 탭의 소자 편집 패널과도 연동됩니다. 모델
라이브러리는 `10_AI_Project\lib`에 있습니다.

```text
OPA379 (기본), TLV9041D, TLV9042
```

세 모델의 subckt 핀 순서를 라이브러리에서 직접 대조해 확인했습니다.

| 모델 | subckt 선언 |
| --- | --- |
| OPA379 | `.SUBCKT OPA379 1 3 5 2 4` (주석 `PINOUT ORDER +IN -IN +V -V OUT`) |
| TLV9041D | `.subckt TLV9041D IN+ IN- VCC VEE OUT` |
| TLV9042 | `.subckt TLV9042 IN+ IN- VCC VEE OUT` |

셋 다 `(IN+, IN-, V+, V-, OUT)`로 같으므로 **노드 순서는 바꾸지 않고 모델
이름만 교체**합니다. 기본값이 아닌 모델을 고르면 해당 `.lib`을 netlist에
추가로 include합니다.

교체 대상은 **U3 하나뿐**입니다. 필터부 U1과 U2는 OPA379로 고정입니다.
선택한 모델은 AC, DC 동작점, Active Detector, Transient **네 해석 모두**에
적용됩니다.

그래서 모델을 바꿔도 **AC의 f0와 gain은 그대로입니다.** AC 출력 node인
`/v_filt`는 검출기 앞단이고 필터 opamp는 고정이기 때문입니다. 달라지는 것은
`Venv_DC`와 detector, transient의 envelope입니다. ch00 실측값입니다.

| U3 opamp | AC f0 | AC gain | Venv_DC |
| --- | --- | --- | --- |
| OPA379 | 167.7 Hz | 6.03 dB | 917.83 mV |
| TLV9041D | 167.7 Hz | 6.03 dB | 943.45 mV |
| TLV9042 | 167.7 Hz | 6.03 dB | 948.24 mV |

선택한 모델은 채널별 값이며 DC 탭 회로도와 `고정 소자 / 모델` 표에도 함께
표시됩니다.

주의: TLV9041과 TLV9042는 최소 동작전압이 1.8 V인데 이 회로의 단전원이 정확히
1.8 V입니다. 경계 조건이므로 모델을 바꾸면 `v_env`의 DC 기준점이 이동할 수
있습니다. 실측 결과는 [VALIDATION_ko.md](VALIDATION_ko.md)에 있습니다.

### 5.4 공유 소자값 패널

Active Detector와 Transient 탭은 같은 `공유 소자값` 패널을 씁니다. 편집
항목은 R4, R5, R6, C3, Vthr, U3 opamp이며 DC 탭과 같은 AppState를 갱신하므로
시뮬을 돌리면서 바로 값을 바꿔볼 수 있습니다.

**패널은 한 번에 한 채널만 편집합니다.** 어느 채널인지는 패널 제목에
`공유 소자값 · ch05`처럼 표시되며, **그 탭의 채널 목록에서 채널을 클릭하면
패널 값이 즉시 그 채널로 바뀝니다.**

DC 탭도 같은 방식으로 동작합니다. DC 채널 목록에서 채널을 클릭하면 그 채널의
DC 결과가 있는 경우 회로도와 소자 편집 패널이 함께 그 채널로 바뀝니다. 아직
결과가 없는 채널은 보여줄 회로도가 없으므로 무시합니다.

#### 선택 채널 일괄 적용

패널의 `선택 채널 일괄 적용 (Vthr 제외)` 체크박스를 켜면, 그 탭 채널 목록에서
선택한 모든 채널에 한 번에 적용됩니다. 기본 CSV에서 R4, R5, R6, C3, U3 opamp는
16채널이 모두 같은 값이라 보통 이렇게 쓰는 편이 편합니다.

- 적용 대상: R4, R5, R6, C3, U3 opamp
- 제외: **Vthr는 패널이 보고 있는 채널에만** 적용됩니다. 채널마다 R7/R8로
  따로 맞춘 값이라 일괄 적용하면 그 튜닝이 모두 사라지기 때문입니다.
- 패널 채널이 목록에서 빠져 있어도 항상 포함되므로 입력한 값이 유실되지
  않습니다.
- 여러 채널을 바꿔도 `component_revision`은 **한 번만** 증가합니다.
- 적용 후 패널 상태줄에 `4개 채널에 일괄 적용됨 ch00, ch01, ch02, ch03`처럼
  대상이 표시됩니다.

체크를 끄면 기존처럼 패널이 보고 있는 한 채널만 바뀝니다.

따라서 탭마다 다른 채널을 보고 있을 수 있습니다. 한 탭에서 값을 바꿨는데 다른
탭 패널이 그대로라면 두 패널이 서로 다른 채널을 보고 있다는 뜻이고, 이때
패널 상태줄에 `ch05가 다른 탭에서 변경됨 (이 패널은 ch00)`이라고 이유가
표시됩니다. 같은 채널을 보고 있으면 편집이 즉시 서로 반영됩니다. 전 채널의
현재 값을 한눈에 비교하려면 `소자값 보기`를 쓰면 됩니다. Vthr를 입력하면 R7과 R8이 다시
계산되고 DC 탭의 Vmargin도 함께 바뀝니다.

Active Detector에서 R4, R5, R6는 kohm 단위로 편집합니다. R4 또는 R5를
바꾸면 R6 병렬값을 소수점 둘째자리까지 자동 입력합니다. R6 직접 편집은
R4와 R5를 바꾸지 않으며, 이후 R4 또는 R5를 다시 바꾸면 R6를 다시 계산합니다.

C3는 read only combobox이며 다음 nF 값만 선택할 수 있습니다.

```text
1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6,
6.8, 8.2, 10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100
```

DC와 Active Detector는 같은 AppState의 Channel 객체를 사용합니다. 한 탭의
유효한 편집이 확정되면 다른 탭 입력란을 즉시 갱신합니다. 수정한 탭에는
`변경됨`, 다른 탭에는 `다른 탭에서 변경됨`을 표시합니다. 동기화 callback은
새 편집으로 처리하지 않으므로 사용자 편집 한 번에 `component_revision`이
한 번만 증가합니다. 값 동기화만으로 simulation을 자동 시작하지 않습니다.

## 6. Transient 시뮬

Transient는 WAV to PWL 결과를 선택해 여러 채널에 실행하는 기존 workflow를
유지합니다. 입력 source는 V4의 기존 연결을 유지하고 파형만 PWL로 교체합니다.
저장 node와 CSV 열은 다음과 같습니다.

```text
time_s, vin_v, v_filt_v, v_env_v, v_thr_v
```

화면에서는 세 그래프를 사용합니다.

1. `v_in`
2. `v_filt`
3. `v_env + v_thr`

시간은 ms, 전압은 mV입니다. 수동 X범위와 각 그래프의 수동 Y범위, 커서,
결과 PWL, 결과 채널은 채널 변경과 재실행 뒤에도 유지됩니다. Detector 실행은
이 상태를 변경하지 않습니다.

세 번째 그래프의 자동 Y범위는 **`v_env`만 기준**으로 정합니다. `v_thr`은 거의
DC 상수라 함께 넣으면 범위를 지배해 `v_env` 파형이 눌립니다. `v_thr`이 범위를
벗어나면 벗어난 쪽 축 경계에 점선으로 붙여 그리고
`v_thr = 917.46 mV ↑ (범위 밖)`처럼 실제 값을 함께 표시합니다.

### 6.1 실행 중 결과 확인

여러 채널을 선택해 실행하면 각 채널이 끝나는 즉시 결과를 볼 수 있습니다.
전체 job이 끝날 때까지 기다리지 않습니다.

- 첫 번째로 끝난 job의 그래프가 바로 표시됩니다.
- 이후 끝나는 채널은 `채널 ch` 버튼이 하나씩 활성화되며, 누르면 그 채널의
  그래프로 바뀝니다.
- 상태줄에 `Transient 2/8 jobs 완료`처럼 진행 상황이 표시됩니다.

실행 중 나중에 끝난 job이 지금 보고 있는 그래프를 가로채지 않습니다. 첫
결과만 자동으로 그리고, 그 뒤부터는 사용자가 버튼을 눌러야 바뀝니다. 그래서
실행 중에 그래프를 확대하거나 커서를 찍어 두어도 남은 job이 끝날 때 사라지지
않습니다.

기본 시간 설정은 output step 10 us, maximum step 5 us이며, 입력 PWL Vpp는
기본 10 mVpp입니다. 원 PWL은 수정하지 않고 실행용 include만 scaling합니다.

## 7. WAV to PWL 시뮬

WAV는 평균을 빼 DC를 0 V로 만든 뒤 peak to peak 범위를 기본 10 mVpp로
scaling합니다. 단어 폴더 구조를 보존하고 마지막 sample 다음 시각에 0 V tail을
추가합니다. 원 WAV와 기존 PWL을 임의로 바꾸지 않습니다.

## 8. 결과 파일

AC, OP, Detector, Transient는 서로 다른 netlist, rawfile, log, CSV를
사용합니다. 주요 Detector 결과 구조는 다음과 같습니다.

```text
results/detector_run_날짜시간/chXX/
  channel_XX_params.inc
  detector_stimulus.inc
  channel_XX_detector.cir
  detector.raw
  detector.csv            (선택, 기본 생성 안 함)
  detector_settings.csv
  detector_ngspice.log
  detector_launcher.log
```

세션 상위에는 실행 시점 소자값을 기록한 `applied_components.csv`가 있습니다.

### 8.1 결과 CSV 선택 생성

`detector.csv`와 `transient.csv`는 각각 `detector.raw`, `tran.raw`와 내용이
같은 중복본이며 한 job에서 수십 MB가 됩니다. 그래서 GUI 기본값은 생성하지
않음이고, `Transient 설정`과 `Gated sine 설정`의 체크박스로 켤 수 있습니다.

CSV를 꺼도 rawfile, `detector_settings.csv`, `applied_components.csv`, 로그는
그대로 생성되며 그래프, 커서, Headroom/Rise/Fall 측정도 전혀 영향받지
않습니다. 결과는 rawfile에서 다시 읽으므로 손실되는 정보가 없습니다.

Python API에서는 `TransientSettings(write_csv=...)`와
`DetectorSettings(write_csv=...)`로 지정하며, 기존 호출 호환을 위해 라이브러리
기본값은 생성함입니다.

### 8.2 Windows 경로 길이 제한

Transient 결과는 다음 구조를 사용합니다.

```text
results/tran_run_날짜시간/stim_0001_단어/chXX/
```

`ngspice_con.exe`는 일반 Win32 프로그램이라 경로가 260자를 넘으면 파일을 열지
못하고 `No such file or directory`만 남깁니다. Python은 같은 경로를 문제없이
만들기 때문에 원인을 알아보기 어렵습니다.

그래서 두 가지를 적용했습니다.

1. stimulus 폴더 이름은 `stim_0001_단어` 형태로 길이가 고정됩니다. PWL 파일명
   전체는 폴더 이름에 넣지 않고 `transient_summary.csv`와 생성된 stimulus
   include 머리말에 보존합니다.
2. 실행 직전에 생성될 경로 길이를 검사해, 한계를 넘으면 ngspice를 실행하지 않고
   초과 글자 수와 해당 경로를 알려줍니다.

경로 초과 오류가 나오면 GUI의 `결과 폴더`를 짧은 경로(예: `C:\ngspice_results`)로
바꾸면 해결됩니다.

### 8.3 실행 실패 진단

AC/OP, Transient, Active Detector는 실패 시 같은 형식의 진단을 표시합니다.

1. 실패 요약 (종료 코드, 또는 코드 0인데 rawfile 없음)
2. 오류 행과 원본 행번호. ngspice 로그를 먼저 보고, 거기에 없으면 launcher
   로그를 봅니다. 오류 행이 없으면 경고 행을 대신 표시합니다.
   ngspice가 자기 `-o` 로그조차 열지 못한 경우에는 오류가 child stdout,
   즉 launcher 로그에만 남기 때문에 두 로그를 모두 확인해야 합니다.
3. ngspice와 launcher 로그의 마지막 부분
4. 확인할 점 (PSpice 호환 초기화, 모델 include 경로)
5. 두 로그 파일의 전체 경로

ngspice는 netlist와 model 오류를 로그 앞부분에 기록하므로, 마지막 부분만
보면 원인을 놓칩니다. 그래서 위치와 무관하게 오류 행을 따로 추출합니다.

## 9. 검사와 실제 모델 실행 제한

전체 회귀검사는 다음 명령으로 실행합니다.

```powershell
py -m unittest discover -s tests -v
```

검사는 142개이며 기존 v14 59개 항목과 다음 기능을 포함합니다.

- dB와 Linear 변환, 작은 Linear 값 보존, 단위별 수동 범위
- 최신 net 이름의 netlist, parser, CSV
- `/vin` gated sine과 전체 회로 보존
- Detector와 PWL Transient 상태 독립성
- Gate 이전 DC 제거
- Vlow와 Vhigh 기반 Headroom, Rise, Fall, 선형 보간
- 잘못된 level과 교차 미검출
- R4, R5, R6, C3 탭 동기화와 revision 1회 증가
- stale 결과 표시
- 새 함수와 class의 docstring
- 기존 공개 import와 `SweeperGUI` 메서드 호환
- raw parser의 실수/복소수 표현, Fortran `D` 지수 fallback, 잘린 파일 거부
- 결과 CSV 선택 생성과 기존 라이브러리 기본값
- 세 실행 경로 공통 실패 진단
- U3 opamp 교체 시 노드 순서 보존과 라이브러리 include
- 소자 버전 CSV의 opamp 열 왕복
- 회로도 overlay 좌표가 자기 자리표시자 안에 들어가는지

실제 OPA379, LPV7215, BAT54WT1 모델을 사용하는 Windows ngspice 해석은
사용자가 확인한 뒤 실행해야 합니다. 확인 전에는 netlist 생성, parser,
합성 파형, fake ngspice 검사까지만 수행합니다.

## 10. 코드 구조

| 경로 | 역할 |
| --- | --- |
| `sweeper/constants.py` | 최신 net, 파일명, 기본값, C3 허용값 |
| `sweeper/models.py` | Channel, AC/Transient/Detector 설정과 결과 객체 |
| `sweeper/components.py` | CSV 보호, 버전 저장, 공통 소자 편집 |
| `sweeper/values.py` | dB/Linear, 단위, 저항, 범위 계산 |
| `sweeper/netlists.py` | AC, OP, Transient, Detector netlist 생성 |
| `sweeper/results.py` | raw parser, AC metric, Detector 측정, CSV |
| `sweeper/simulation.py` | 독립 simulation runner와 result 생성 |
| `sweeper/gui/state.py` | AppState, revision, cache, 수동 범위 |
| `sweeper/gui/dc_tab.py` | 최신 회로도와 공통 소자 편집 |
| `sweeper/gui/ac_tab.py` | AC dB/Linear 그래프와 metric |
| `sweeper/gui/detector_tab.py` | Active Detector 전용 controller |
| `sweeper/gui/transient_tab.py` | PWL Transient 전용 controller |
| `sweeper/gui/pwl_tab.py` | WAV to PWL controller |
| `sweeper/gui/common.py` | 공통 event와 소자 동기화 |
| `sweeper/gui/app.py` | 다섯 상위 탭과 controller 조정 |

기존처럼 다음 import를 계속 사용할 수 있습니다.

```python
from ngspice_channel_sweeper import Simulator, SweeperGUI
```

Detector 객체도 같은 호환 진입점에서 import할 수 있습니다.
