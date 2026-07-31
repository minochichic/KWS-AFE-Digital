# ngspice 16채널 AC / OP / Transient GUI

`channel_components.csv`의 16채널 소자값을 공통 회로에 적용해 AC Sweep,
DC 동작점(`.op`), 음성 PWL transient를 실행하는 Windows용 GUI입니다.
기본 소자 CSV는 읽기 전용으로 보존하며, 편집값은 버전 CSV와 각 실행의
`applied_components.csv`에 따로 남깁니다.

함수별 역할과 데이터 흐름은
[`ARCHITECTURE_ko.md`](ARCHITECTURE_ko.md)에 정리되어 있습니다.

## 1. 준비와 실행

1. Windows ngspice를 설치하고 `bin` 폴더를 PATH에 등록합니다.
   자동화에는 `ngspice.exe`보다 콘솔용 `ngspice_con.exe`를 우선 사용합니다.
2. `netlist_template.cir`의 BAT54WT1, LPV7215, OPA379 `.include` 경로를
   실제 PC 경로와 맞춥니다.
3. 패키지를 설치하고 GUI를 실행합니다.

```powershell
py -m pip install -r requirements.txt
py ngspice_channel_sweeper.py
```

또는 `run_gui.bat`을 더블클릭합니다.

GUI는 서로 독립적인 네 상위 탭으로 구성됩니다.

- `DC 동작점` — 초기 탭
- `AC 시뮬`
- `Transient 시뮬`
- `WAV → PWL 시뮬`

## 2. DC 동작점 / AC 시뮬

상단에서 ngspice, 공통 네트리스트, 결과 폴더와 채널을 선택합니다.
DC와 AC는 각각 자기 상위 탭의 실행 버튼으로 시작합니다. 두 해석은
netlist/rawfile/CSV가 분리되어 있고 한 해석을 실행해도 다른 결과 화면을
지우거나 현재 보고 있는 상위/AC 결과 탭으로 강제 이동하지 않습니다.

- 기본 AC: 100 points/decade, 10 Hz~20 kHz
- 기본 AC 출력: `/v_filt_out`
- OP: `.save all`과 `.op`으로 모든 전압 노드를 저장
- 채널별 결과:
  `results/run_날짜시간/chXX/ac.csv`, `ac_metrics.csv`,
  `operating_point.csv`
- 세션 결과: `applied_components.csv`, `ac_metrics_summary.csv`

`DC 동작점` 상위 탭은 회로도/노드표/소자 편집기를 표시합니다. `AC 시뮬`
상위 탭 안에는 `AC Magnitude`, `AC Phase`, `실행 로그` 결과 탭이 있습니다.
실행 실패 때도 현재 탭을 유지하고 오류 팝업과 로그만 갱신합니다.

### AC 그래프

Magnitude 오른쪽에는 16채널의 측정 중심주파수, 피크 Gain, Q가 한 줄씩
표시됩니다. 여러 행을 클릭하면 각 중심주파수에 수직선, 피크 점,
`f0/Gain/Q` 주석이 함께 나타납니다. 선택 행 또는 전체 16행을 탭 구분
텍스트로 복사할 수 있습니다.

피크는 decade 스윕의 간격에 맞춰 `log10(f)` 축에서 주변 3점을 2차
보간합니다. 피크보다 \(10\log_{10}2=3.0103\) dB 낮은 양쪽 교차점도
로그 주파수에서 보간하고,

\[
Q=\frac{f_0}{f_H-f_L}
\]

로 계산합니다. 양쪽 교차점이 스윕 범위에 없으면 잘못된 Q를 만들지 않고
`측정 범위 부족`으로 표시합니다.

Magnitude와 Phase의 X축에서 `Linear/Decade`를 선택할 수 있습니다.
Magnitude Y축은 `Linear (dB)/Decade (|V|)`를 지원합니다. Decade Y축은
dB를 다시 로그축에 놓지 않고 \(|V|=10^{G_\mathrm{dB}/20}\)로 복원합니다.
Magnitude는 X/Y 최소·최대 범위를 직접 입력할 수 있고, DC를 실행한 뒤
AC를 다시 실행해도 그 입력값을 유지합니다. 자동 Y범위는 데이터 최소–최대가
축 높이의 90%를 차지하도록 양쪽에 같은 여백을 둡니다. 그래프 값 확인은
평소 마우스 이동으로 시작되지 않습니다. `커서 추가`를 누르면 세로 보조선과
값 표시가 가장 가까운 그래프 선을 따라가고, 원하는 위치를 클릭하면 그
데이터 점에 고정됩니다. 이 동작을 반복해 여러 커서를 남기거나
`커서 전체 삭제`로 지울 수 있습니다.

### DC 동작점 회로도와 소자 편집

회로도와 오른쪽 노드 표는 ngspice 내부 V 값을 `mV = V × 1000`으로
표시합니다. 회로도의 `{R1}`, `{RA}`, `{C1}` 자리에는 해당 실행에 사용한
채널별 kΩ/nF 값이 표시됩니다. 휠로 0.5~2.5배 확대하고 좌클릭 드래그로
이동할 수 있습니다. 채널 변경과 확대/축소도 현재 결과 탭을 유지합니다.

오른쪽 `적용 소자값 편집`에서 RA, R1, R2/R3, R4~R8, C1/C2, C3를
바꿀 수 있습니다. Vref 직접 입력은 제거했고 `목표 V_margin [mV]`을
편집합니다. 최근 DC 결과의 Vdet에 대해

\[
V_\mathrm{margin}=V_\mathrm{ref}-V_\mathrm{det},
\qquad
V_\mathrm{ref}=V_\mathrm{det}+V_\mathrm{margin}
\]

로 목표 Vref를 정합니다. 이 값은 DC 상태에서 detector가 비교기 임계값까지
더 상승해야 하는 전압 여유이므로 목표 Vmargin은 0보다 큰 값만 허용합니다.
회로 연결은 R7=1.8 V측, R8=GND측이므로

\[
V_\mathrm{ref}=1.8\frac{R8}{R7+R8}
\]

입니다. 기존 R7+R8은 분압 전류와 입력 부하 규모가 갑자기 바뀌지 않게 하는
명목 저항으로만 사용하고, 계산한 R7과 R8을 각각 0.01 kΩ로 독립
반올림합니다. 따라서 합계를 정확히 1 MΩ로 맞추는 보정은 하지 않습니다.
반올림된 R7/R8가 만드는 실제 Vref는 읽기 전용 `계산 Vref [mV]`에
표시됩니다.

RA와 R1~R8은 입력·계산·netlist 표시·새 버전 CSV에서 모두 kΩ 단위
소수점 둘째자리까지 사용합니다. R4 또는 R5가 바뀌는 순간에는

\[
R6=R4\parallel R5=\frac{R4R5}{R4+R5}
\]

를 0.01 kΩ로 반올림해 R6 입력칸에 먼저 넣습니다. 이 자동 입력은 기본값
제안이며 제약식이 아니므로, 그 뒤 R6만 다시 입력해 독립된 값으로 사용할 수
있습니다. R4나 R5를 다시 바꾸면 새 병렬값이 다시 R6에 제안됩니다.

그 아래에는 최근 실제 `.op` 결과의

\[
V_\mathrm{margin}=V_\mathrm{ref}-V_\mathrm{det}
\]

을 부호가 있는 mV 값으로 표시합니다. detector 소자도 동시에 바꿔 Vdet가
달라지면 첫 DC 결과의 새 Vdet로 R7/R8을 한 번 다시 계산한 뒤 최종 DC를
자동 재실행합니다. `변경값 적용 + DC 실행` 전후에는 현재 보고 있던
`노드 전압/적용 소자값 편집` 탭, 회로도–편집창 분할폭, 회로도 스크롤
위치와 애플리케이션 창 크기를 유지합니다. 이 상태는 버튼을 누르는 순간
동결해 Vmargin 보정 때문에 DC가 두 번 실행되는 경우에도 그대로 복원합니다.
오른쪽 세부 영역의 초기 폭도 이전보다 조금 줄여 회로도를 더 넓게 보이게
했습니다. AC와 Transient 실행 버튼도
실행 직전에 보이는 편집값을 자동 커밋합니다.
`channel_components.csv`에는 절대 쓰지 않으며:

- `새 버전 저장`: `component_versions/channel_components_v날짜시간.csv`
- 실행 snapshot: 세션의 `applied_components.csv`
- `버전 불러오기`: 저장된 16채널 버전을 메모리에 적용
- `기본값`: 읽기 전용 출고 CSV 복원

V2의 netlist 문장은 `V2 1.8V 0.9V DC 0.9`입니다. 즉 V2 자체의 전압은
0.9 V이고, 음단이 0.9 V rail이므로 양단인 `1.8V` 노드가 이상적으로
1.8 V가 됩니다. 이전 GUI의 `(0.9 V → 1.8 V)`는 이 관계를 설명하려던
표시일 뿐 모델명이 아니며 혼동을 줄 수 있어 `DC 0.9 V`로 단순화했습니다.
V2 소스 자체는 1.8 V rail을 만드는 데 필요하므로 netlist에서는 제거하지
않았습니다.

## 3. WAV → PWL 시뮬

기본 WAV 루트는 다음 경로입니다.

```text
C:\Users\duyou\Desktop\Personal\10_Study\10_SemiContest\10_AI_Project\speech_commands_for_sim
```

루트 아래 `go`, `left`, `yes` 같은 단어 폴더를 재귀적으로 스캔하고,
PWL 출력 아래에 같은 단어/하위 폴더 구조를 만듭니다.

```text
speech_commands_for_sim/
  go/sample.wav
  left/sample.wav
  pwl/
    go/sample.pwl
    left/sample.pwl
    pwl_manifest.csv
```

각 파일의 mono 샘플을 \(x[n]\), 평균을 \(\bar{x}\), 최소·최대를
\(x_\min,x_\max\)라 하면 다음 식을 적용합니다.

\[
y[n]=(x[n]-\bar{x})
\frac{0.010}{x_\max-x_\min}\quad[\mathrm{V}]
\]

이 식은 먼저 평균을 빼 discrete DC 성분을 0 V로 만들고, 그 다음
peak-to-peak 범위를 정확히 0.010 V, 즉 10 mVpp로 맞춥니다. min/max의
중간을 0으로 옮기는 방식과 달리 비대칭 음성에서도 평균 DC가 0이라는
장점이 있습니다. 상수/무음 파일은 10 mVpp를 만들 수 없으므로 0 V로
내보내고 manifest 상태에 표시합니다.

Google Speech Commands v2의 일반적인 mono 16-bit PCM을 지원하며,
8/16/24/32-bit integer PCM과 다채널 WAV도 처리합니다. 다채널은 프레임별
채널 평균으로 mono 변환합니다. 압축 WAV는 명확한 오류로 거부합니다.

원 WAV가 N sample, sample rate가 \(f_s\)이면 원 데이터는
\(t=(0,\ldots,N-1)/f_s\)에 기록됩니다. 추가로 \(t=N/f_s\)에 0 V 한 점을
붙입니다. PWL은 마지막 값을 계속 유지하므로 transient stop time이 WAV보다
길어도 이후 입력은 0 V입니다.

`.pwl`은 회로와 독립적인 두 열 형식입니다.

```text
* ngspice PWL data v1: time_s voltage_v
0              -0.001234
0.0000625       0.002345
...
1               0
```

`pwl_manifest.csv`에는 원본/출력 경로, sample rate, frame 수, 원 min/max/mean,
출력 min/max/mean/Vpp, 상태를 기록합니다.

## 4. Transient 시뮬

1. `Transient 시뮬` 탭에서 변환된 PWL 폴더를 선택합니다.
2. 재귀적으로 발견된 PWL 중 실행할 파일을 복수 선택합니다.
3. 실행 채널을 하나 이상 선택합니다.
4. 시간 설정 후 `Transient 실행`을 누릅니다.

실수로 대량 작업을 시작하지 않도록 첫 PWL과 첫 채널만 기본 선택합니다.
버튼에는 `파일 수 × 채널 수`의 예상 job 수가 표시됩니다.

시간 설정:

- `입력 PWL Vpp [mVpp]`: 기본 `10`, 원하는 양의 Vpp로 변경 가능
- `output step tstep [s]`: 기본 `10u` = \(10\,\mu s\)
- `stop time [s]`: 빈칸이면 PWL의 마지막 0 V 시각
- `maximum step tmax [s]`: 기본 `5u` = \(5\,\mu s\)

입력 Vpp를 바꿔도 원본 `.pwl` 파일은 수정하지 않습니다. 실행 직전에
원본 PWL의 peak-to-peak 값을 \(V_{\mathrm{pp,src}}\)로 측정하고 모든 전압에
\(V_{\mathrm{pp,target}}/V_{\mathrm{pp,src}}\)를 곱한 실행용 include를
만듭니다. 따라서 파형 모양, 0 V DC, 마지막 0 V tail은 그대로 유지됩니다.
상수/무음 PWL은 스케일할 신호가 없으므로 계속 0 V입니다.

시간 입력은 `1e-5`뿐 아니라 `10u`, `10us`, `2.5ms`, `1s` 같은 표기를
받습니다. 따라서 이전 기본값 `1e-5 s`는 `10u`와 정확히 같은 값이며,
`1e5 s` 또는 \(10^5\)초가 아닙니다. 1초 해석이라면 \(1/10\,\mu s=100{,}000\)
개의 명목 간격에 해당합니다.

[ngspice 공식 정의] `tstep`은 출력/plotting 간격이며 post-processor에서는 제안
계산 간격이고, `tmax`는 solver가 넘지 못하는 최대 내부 시간간격입니다.
[설계 기준/추론] 일반적인 1차 기준은 최고 관심주파수 \(f_\max\)의 한 주기를 최소 20점으로
나누는 것입니다.

\[
t_\max \le \frac{1}{20f_\max}
\]

현재 최고 채널 6.761 kHz에는 \(t_\max\le7.40\,\mu s\), 16 kHz WAV의
Nyquist 8 kHz까지 보존하려면 \(t_\max\le6.25\,\mu s\)입니다. 현재 버전은
5 µs를 보수적인 기본 `tmax`로 사용합니다. `tstep=10 µs`는 1초당 약
100,000간격으로 표시 데이터량을 제한하기 위한 절충입니다. comparator
edge 시각을 수 µs 수준으로 더 정밀하게 볼 때는 `tstep=tmax=2u~5u`,
빠른 확인에는 `10u` 정도가 이 회로에서 현실적입니다. 이 값은 모든 회로의
보편 상수가 아니며, 더 빠른 입력 edge나 더 짧은 time constant가 있으면
그 기준의 1/10~1/20보다도 작게 잡아 수렴 결과를 비교해야 합니다.

예를 들어 stop time을 1.5로 입력하면 1초 음성이 끝난 뒤 나머지 0.5초는
PWL의 마지막 0 V가 유지됩니다.

도구는 공통 netlist에서 `/vin`에 연결된 독립 전압원을 찾습니다. 이름이
`V4`든 `Vsin`이든 기존 소스명, 양단 노드, 극성을 보존하고 SIN/AC 정의를
PWL source include로 교체합니다. AC/OP netlist는 별도로 생성되므로 이
교체의 영향을 받지 않습니다.

Transient netlist는 다음 노드를 저장합니다.

- `/vin` → `Vin`
- `/v_filt_out` → `Vfilt_out`
- `/v_detect_out` → 그래프의 `Venv_out`
- `/v_ref` → `Vref`

결과 화면에는 요청한 세 그래프가 기본으로 함께 나타납니다.

1. Vin
2. Vfilt_out
3. Venv_out + Vref

`Venv_out > Vref` 음영, 구간 판정, 구간 CSV는 제거했습니다. 세 번째
그래프에는 두 아날로그 곡선만 표시됩니다.

실행이 끝나면 결과 선택은 `결과 PWL` 드롭다운과 `ch00~ch15` 선택줄로
분리됩니다. PWL을 고르면 그 PWL로 실제 완료된 채널만 활성화되고, 활성
채널을 클릭하면 즉시 해당 그래프가 표시됩니다. 같은 PWL의 여러 채널을
비교할 때 아래의 수동 X/Y 범위는 그대로 유지됩니다.

Transient 상단에는 현재 소자값이 기본 CSV와 다른 채널을 표시합니다.
표시 중인 결과가 현재 소자값으로 실행된 것인지, DC 편집 전 결과여서
재실행이 필요한지도 함께 표시합니다. DC 편집칸에 입력했지만 아직
`변경값 적용 + DC 실행`을 누르지 않은 값도 별도로 알립니다. DC에서 바꾼
값은 다음 Transient 실행 직전에 자동 적용됩니다.

### Transient 그래프 조작

- X축은 ms, 모든 Y축은 mV로 표시합니다.
- rawfile과 `transient.csv`는 계산용 SI 단위(s, V)를 그대로 보존합니다.
- 자동 Y범위는 각 그래프 데이터의 최소–최대가 축 높이의 정확히 90%를
  차지하도록 위·아래에 같은 여백을 둡니다. 세 번째 그래프는 Venv_out과
  Vref를 합친 최소–최대를 사용합니다.
- 범위 조절부는 한 줄의 `X [ms]` 그룹과 `Y [mV]` 그룹으로 압축했습니다.
  상태 안내도 그 아래 한 줄만 사용해 그래프의 세로 공간을 더 확보합니다.
- X축은 `자동/수동` 모드와 `min/max`, `적용`을 사용합니다.
  X 수동범위는 세 그래프가 공유합니다.
- Y축은 그래프 대상을 고른 뒤 그 대상의 `자동/수동` 모드와
  `min/max`, `적용`을 사용합니다. 세 Y 그래프는 모드와 수동값을
  각각 독립적으로 기억합니다.
- 한 축을 자동으로 바꿔도 다른 축의 수동값은 해제되지 않습니다. 수동
  X범위와 그래프별 Y범위는 PWL/채널 변경, DC 소자값 변경, Transient
  재실행 뒤에도 유지됩니다.
- DC 탭에서 돌아왔을 때 기존 Transient 결과가 빈 화면으로 보이지 않도록
  캔버스만 다시 그립니다. 결과 선택과 저장된 수동 범위는 바꾸지 않습니다.
- 느린 마우스 구간 드래그와 사각 zoom 도구막대는 제거했습니다. 세 그래프
  모두 숫자 X 범위를 공유하고, Y 범위는 선택한 그래프에만 적용됩니다.
- 평소 마우스 이동에는 커서가 나타나지 않습니다. `커서 추가`를 누른 다음
  움직이면 세로 보조선·점·ms/mV 값이 가장 가까운 선을 따라갑니다. 클릭한
  위치는 고정되며 반복해서 여러 커서를 남길 수 있습니다. 이동 중에는
  blitting으로 커서만 다시 그려 큰 transient 데이터의 전체 재렌더링을
  피합니다.

Transient 결과 구조:

```text
results/tran_run_날짜시간/
  applied_components.csv
  transient_summary.csv
  stim_0001_단어_파일/
    stimulus.pwl.inc
    ch00/
      channel_00_params.inc
      channel_00_tran.cir
      tran.raw
      transient.csv
      tran_ngspice.log
      tran_launcher.log
```

`transient.csv`에는 time, Vin, Vfilt_out, Venv_out, Vref만 기록합니다.

## 5. ngspice 실행과 로그

AC, OP, transient는 ngspice 공식 배치 rawfile 옵션을 사용합니다.

```text
ngspice_con.exe -b -o ac_ngspice.log   -r ac.raw   channel_XX_ac.cir
ngspice_con.exe -b -o op_ngspice.log   -r op.raw   channel_XX_op.cir
ngspice_con.exe -b -o tran_ngspice.log -r tran.raw channel_XX_tran.cir
```

`.spiceinit`의 `set filetype=ascii`로 rawfile을 Python에서 직접 읽습니다.
`set ngbehavior=pski`는 PSpice 모델과 `/node` 형식의 KiCad 노드를
netlist를 읽기 전에 활성화합니다. `.control/wrdata`는 사용하지 않습니다.

ngspice 46 공식 매뉴얼은 independent PWL source의 time/value 쌍 사이를
선형 보간한다고 설명하고, transient 형식을
`.tran tstep tstop <tstart <tmax>>`로 정의합니다.

- [ngspice 46 공식 매뉴얼](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf)

실행 실패 시 각 channel 폴더의 `*_ngspice.log`, `*_launcher.log`와 세션
`launcher.log`를 확인합니다. 종료코드가 0이어도 rawfile이 없으면 실패로
처리합니다.

## 6. 명령행과 회귀검사

16채널 AC/OP netlist 생성만 검사:

```powershell
py ngspice_channel_sweeper.py --validate-only
```

GUI 없이 AC/OP 실행:

```powershell
py ngspice_channel_sweeper.py --headless --channels 0,3,7 --analysis ac
py ngspice_channel_sweeper.py --headless --channels all --analysis both
```

Transient와 WAV 변환은 현재 GUI에서 실행합니다.

전체 회귀검사:

```powershell
py -m unittest discover -s tests -v
```

또는 `run_tests.bat`을 실행합니다. 검사는 기본 CSV 보호, 버전 왕복,
AC f0/Q, mV 표시, 네 상위 탭 순서, Vmargin→Vref→0.01 kΩ R7/R8 계산,
모든 저항의 소수점 둘째자리 처리, R4/R5→R6 병렬값 자동 제안, WAV
정규화, 단어 폴더 변환, 마지막 0 V, Vsin/V4 교체, transient raw 파싱,
가변 입력 Vpp, 90% 자동범위, 축별 자동/수동 범위 유지, PWL+16채널
결과 선택, 소자값 변경/결과 반영 상태, 버튼식 선 추종·다중 고정 커서,
DC 세부 탭/분할폭/스크롤 동결·복원, Transient 재도색,
가짜 ngspice 실행→CSV 경로와 모든 함수/class의 docstring 누락 여부를
확인합니다.

## 7. 다음 확장 지점

Transient 구현은 `TransientSettings`, `TransientSimulator`,
`read_transient_raw()`, 전용 결과 객체와 GUI renderer로 AC/OP에서
분리했습니다. 이후 10 ms frame 스펙트로그램, 16채널 feature vector,
detector G/τ 및 offset sweep을 추가할 때 `transient.csv`를 입력으로 쓰거나
새 post-processing 모듈을 연결할 수 있습니다. ngspice process 실행,
기본 CSV 보호와 네 상위 탭 상태에는 변경이 필요 없습니다.
