# Vivado / XSim 작업 안내서

이 문서는 현재 검증 모델 `bd_base_ft20_partial75`와 FPGA top
`kws_stream_top`을 기준으로 한다. 모든 명령은 저장소 루트의 PowerShell에서
실행한다.

```powershell
cd C:\Users\okbong\repos\KWS-AFE-Digital-windowing
```

## 1. 검증 순서

1. `xvlog`: Verilog 문법과 모듈을 분석한다.
2. `xelab`: 파라미터를 적용하고 실제 시뮬레이션 계층을 만든다.
3. `xsim`: RTL 동작을 골든 벡터와 비교한다.
4. `synth_design`: RTL을 FPGA LUT, FF, RAM, DSP로 합성한다.
5. `place_design` / `route_design`: 실제 FPGA 위치와 배선을 결정한다.
6. `report_timing_summary`: 배선 후에도 50 MHz를 만족하는지 확인한다.
7. `write_bitstream`: 핀 제약과 DRC까지 통과한 구현으로 bitstream을 만든다.

RTL XSim 통과는 논리 동작을, 합성은 FPGA 자원으로 변환 가능함을, 구현은 실제
배치배선과 최종 타이밍을 검증한다. 셋은 서로 대체하지 않는다.

## 2. XSim 배치 실행

PowerShell 실행 정책 때문에 `.ps1`이 차단되면 아래처럼
`-ExecutionPolicy Bypass`를 현재 프로세스에만 적용한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\rtl\run_xsim.ps1 `
  -Name top `
  -Tag bd_base_ft20_partial75
```

주요 테스트는 다음과 같다.

| 이름 | 검증 대상 |
|---|---|
| `capture` | comparator 동기화와 10 ms sticky OR 포착 |
| `window` | 원형 history, snapshot 복사, padding, 연속 포착 |
| `vote` | N회 연속, integer margin, cooldown, gap 처리 |
| `block` | residual block의 두 sub-block, skip, add threshold |
| `plane` | plane buffer가 residual block에 공급하는 frame/flush |
| `tail` | Conv2 pointwise부터 pooled logits와 margin까지 |
| `top` | 저장된 AFE 입력부터 최종 class/score까지 전체 네트워크 |

여러 테스트를 순서대로 실행하려면:

```powershell
foreach ($name in 'capture','window','vote','block','plane','tail','top') {
  powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\rtl\run_xsim.ps1 `
    -Name $name `
    -Tag bd_base_ft20_partial75
}
```

성공 기준은 각 로그 끝의 `0 failures`와 스크립트의 `PASS`다. 결과는
`out/xsim/<name>/`에 저장된다.

```powershell
Get-Content .\out\xsim\top\xsim.log -Tail 20
```

## 3. XSim GUI와 파형

live GUI를 열려면 `-Gui`를 붙인다. GUI에서는 자동으로 `run all`을 하지 않으므로
파형을 고른 뒤 직접 실행할 수 있다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\rtl\run_xsim.ps1 `
  -Name window `
  -Tag bd_base_ft20_partial75 `
  -Gui
```

XSim Tcl Console에서 자주 쓰는 명령:

```tcl
# 현재 scope의 신호 확인
get_objects /tb_window/dut/*

# DUT 바로 아래 신호 추가
add_wave [get_objects /tb_window/dut/*]

# DUT 아래 모든 하위 계층까지 추가
add_wave -r [get_objects /tb_window/dut/*]

# 표시 형식 지정
add_wave -radix hex [get_objects /tb_window/dut/out_frame]

# WDB에는 기록하지만 Wave 창에는 자동 추가하지 않음
log_wave -r [get_objects /tb_window/dut/*]

# 실행 제어
run 1 us
run all
restart
```

큰 `top` 전체를 재귀적으로 추가하면 WDB와 GUI가 매우 무거워진다. 처음에는 다음
경계 신호만 보는 편이 낫다.

```tcl
add_wave [get_objects /tb_top/clk]
add_wave [get_objects /tb_top/rst_n]
add_wave [get_objects /tb_top/start]
add_wave [get_objects /tb_top/iv]
add_wave [get_objects /tb_top/rdy]
add_wave [get_objects /tb_top/dut/st]
add_wave [get_objects /tb_top/dut/class_valid]
add_wave [get_objects /tb_top/dut/class_idx]
add_wave [get_objects /tb_top/dut/score_valid]
add_wave [get_objects /tb_top/dut/keyword_margin]
run all
```

이미 만들어진 snapshot과 WDB를 다시 열려면 저장소 루트에서 다음과 같이 실행한다.

```powershell
xsim tb_top_sim --gui `
  --wdb .\out\xsim\top\tb_top_sim.wdb `
  --autoloadwcfg
```

snapshot이 없거나 소스가 바뀌었다면 이 명령 대신 `run_xsim.ps1 -Gui`로 다시
elaborate한다.

## 4. 합성

현재 스트리밍 top과 export를 합성하는 표준 명령:

```powershell
vivado -mode batch -source rtl/build.tcl `
  -tclargs `
  -tag bd_base_ft20_partial75 `
  -top kws_stream_top
```

출력은 `out/synth/xc7s75fgga484-1/`에 생긴다.

| 파일 | 의미 |
|---|---|
| `post_synth.dcp` | 합성된 netlist checkpoint |
| `utilization.rpt` | 계층별 LUT/FF/RAM/DSP 사용량 |
| `timing_synth.rpt` | 합성 추정 타이밍 |
| `inc/` | 합성 run에서 `$readmemh`가 읽는 ROM 사본 |

PowerShell에서 핵심 결과만 확인하려면:

```powershell
Select-String `
  -Path .\out\synth\xc7s75fgga484-1\timing_synth.rpt `
  -Pattern 'WNS\(ns\)','All user specified timing constraints are met','Clock Summary' `
  -Context 0,10

Select-String `
  -Path .\out\synth\xc7s75fgga484-1\utilization.rpt `
  -Pattern '^\| kws_stream_top','u_window','u_net'
```

`WNS > 0`, `TNS = 0`이면 지정된 합성 타이밍을 만족한다. 이것은 배선 전 추정치이므로
최종 판정은 post-route 보고서로 한다.

## 5. 합성 결과를 Vivado GUI에서 보기

바로 checkpoint를 열려면:

```powershell
vivado -mode gui -source rtl/open_checkpoint.tcl `
  -tclargs out/synth/xc7s75fgga484-1/post_synth.dcp
```

GUI Tcl Console에서 유용한 명령:

```tcl
report_utilization -hierarchical
report_timing_summary
report_drc
report_methodology

# 계층 찾기
get_cells -hier -filter {NAME =~ *u_window*}
get_cells -hier -filter {NAME =~ *u_net*}

# 선택 계층 schematic 열기
show_schematic [get_cells -hier -filter {NAME =~ *u_window*}]
show_schematic [get_cells -hier -filter {NAME =~ *u_net/u_top*}]
```

GUI에서는 Netlist 창에서 인스턴스를 선택하고 **Schematic**을 눌러도 된다. 합성
schematic의 `FDCE`, `FDRE`, `LUT6`, `RAM64M`, `RAMB18E1`, `DSP48E1` 의미는
`docs/fpga_primitives.md`를 참고한다.

## 6. 배치배선과 bitstream

합성 checkpoint가 이미 있으면 다음 명령이 권장된다. 합성을 반복하지 않고
`opt_design`, `place_design`, `route_design`, post-route 보고서, DRC, bitstream 생성을
수행한다.

```powershell
vivado -mode batch -source rtl/implement_checkpoint.tcl `
  -tclargs out/synth/xc7s75fgga484-1/post_synth.dcp
```

소스부터 implementation까지 완전히 다시 실행하려면 다음 명령을 사용한다.

```powershell
vivado -mode batch -source rtl/build.tcl `
  -tclargs `
  -tag bd_base_ft20_partial75 `
  -top kws_stream_top `
  -impl
```

이 단계는 모든 top port의 `PACKAGE_PIN`과 `IOSTANDARD`가 확정됐을 때 실행한다.
핀 하나라도 빠지면 `write_bitstream` 앞 DRC가 실패하는 것이 정상이다. 결과는 다음
파일에서 확인한다.

```text
out/synth/xc7s75fgga484-1/timing_impl.rpt
out/synth/xc7s75fgga484-1/utilization_impl.rpt
out/synth/xc7s75fgga484-1/drc_impl.rpt
out/synth/xc7s75fgga484-1/post_route.dcp
out/synth/xc7s75fgga484-1/kws_stream_top.bit
```

post-route GUI:

```powershell
vivado -mode gui -source rtl/open_checkpoint.tcl `
  -tclargs out/synth/xc7s75fgga484-1/post_route.dcp
```

## 7. 현재 경고 해석

### `VRFC 10-3645: port 'score_valid' remains unconnected`

`kws_top`에 streaming voter용 `score_valid`, `keyword_idx`, `keyword_margin` 출력이
추가됐지만 예전 `tb_top`이 class 출력만 연결해서 발생했다. DUT 계산 오류는 아니지만
interface 변화가 테스트벤치에 반영되지 않았다는 뜻이다. 현재 `tb_top`에서 세 출력을
명시적으로 연결해 제거했다.

### `Synth 8-3848: Net t_rom ... does not have driver`

`kws_pw_conv`의 일부 인스턴스는 `T_FILE=""`이다. 이들은 thresholded bit가 아니라
정수 accumulator를 residual add 또는 affine에 전달하는 `epilogue none` 경로다.
따라서 threshold ROM을 의도적으로 읽지 않고, 경고가 가리키는 `out_frame`도 상위에서
사용하지 않는다. `acc_valid/acc_ch/acc_out`만 유효하다.

현재 파라미터에서 이 경고는 예상된 것이다. `T_FILE`이 비어 있지 않은 인스턴스에서
같은 경고가 나오거나 `$readmemh ... read successfully`가 사라지면 ROM 경로 오류로
간주해야 한다.

### `Synth 8-7137: s1_frame_reg has both Set and reset with same priority`

residual block이 sub0의 실제 출력과 drain용 zero frame을 비동기 reset FSM 안에서
선택해 저장하자 Vivado가 bit별 set/reset 제어로 해석하면서 발생했다. FPGA FF에는 동일
우선순위의 set/reset 조합이 직접 존재하지 않아 합성/RTL simulation 불일치 가능성을
경고한 것이다. 실제 출력이 flush보다 우선한다는 규칙은 유지하되, `s1_frame` 저장을
쓰기 enable만 가진 별도 동기 레지스터로 분리했다. FSM은 push와 상태만 제어한다.
`block`과 `plane` 골든 XSim이 각각 512 frames, 0 failures로 통과했고 재합성에서 이
경고가 사라졌다. AMD UG901도 하나의 FF에 set과 reset을 함께 기술하지 말고 명확한
순차 mux/enable 형태를 사용하도록 권고한다.

### `Synth 8-6014: phase_cyc_reg / st_q_reg / pd_seen_reg was removed`

이 신호들은 `KWS_ASSERT` watchdog과 내부 순서 assertion을 위한 디버그 상태다.
`run_xsim.ps1`은 `KWS_ASSERT`를 정의하므로 XSim에서 사용되지만, `build.tcl`의 하드웨어
합성은 이를 정의하지 않아 관측자가 사라진다. Vivado가 기능 출력에 영향이 없는
레지스터를 제거한 정상 최적화다.

### 합성 중 `PACKAGE_PIN is not supported for elaborated designs`

핀 위치는 implementation 속성이라 RTL elaboration 단계에서는 적용되지 않는다.
현재 스크립트는 합성 netlist를 만든 뒤 XDC를 다시 적용한다. 로그 후반의
`applied 25 pin constraints`와 implementation DRC를 확인해야 한다. 앞부분의
`applied 0 pin constraints`만 보고 핀이 사라졌다고 판단하면 안 된다.

### implementation DRC: `CFGBVS-1`, `DPIP-1`, `REQP-1840`

- `CFGBVS-1`: configuration bank 0의 실제 전압과 CFGBVS 결선 정보가 XDC에 없다.
  보드 회로도에서 확인한 뒤 `CFGBVS`와 `CONFIG_VOLTAGE`를 지정한다. 값을 추측해
  경고만 숨기지 않는다.
- `DPIP-1`: DSP48 입력에 pipeline register를 넣으면 최대 주파수를 높일 수 있다는
  성능 권고다. 현재 50 MHz post-route WNS가 양수이므로 기능상 blocker가 아니다.
- `REQP-1840`: affine 계수 BRAM의 읽기 주소/enable을 만드는 레지스터가 비동기 reset
  assert를 가진다는 경고다. 내부 reset은 동기 deassert되고 reset 중 계산 결과를 쓰지
  않으므로 현재 reset 계약에서는 허용한다. 보드 smoke test에서 reset 직후 첫 추론을
  반드시 확인한다.
- `CHECK-3`: `REQP-1840` 표시 개수가 report 한도에 도달했다는 안내다.

## 8. 현재 검증 기준선

`bd_base_ft20_partial75`, `kws_stream_top`, Vivado 2026.1 기준:

- `capture`: 3 frames, 0 failures
- `window`: 33 frames, 0 failures
- `vote`: 3 scenarios, 0 failures
- `block`: 512 frames, 0 failures
- `plane`: 512 frames, 0 failures
- `tail`: 8 frames, 0 failures
- `top`: 8 clips, 0 failures
- synthesis clock: 50 MHz
- synthesis timing: WNS +6.003 ns, TNS 0
- synthesis resource: LUT 20,375, FF 16,607, RAMB18 6, DSP 7
- sliding buffers: `history`와 `snapshot` 각각 `128 x 16` 분산 RAM
- post-route timing: WNS +4.600 ns, TNS 0, WHS +0.024 ns, THS 0
- post-route resource: LUT 20,148 (41.98%), FF 16,611 (17.30%),
  RAMB18 6 (BRAM tile 3), DSP 7
- routing: unrouted nets 0, node overlaps 0
- bitstream DRC: 0 errors; `kws_stream_top.bit` 생성

다음 검증 단계는 연속 comparator 입력부터 voter event까지 한 번에 확인하는
`tb_stream_top` 통합 테스트와 실제 보드 smoke test다.
