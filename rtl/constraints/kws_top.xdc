# 제약 파일 — kws_top / kws_frame_ctrl
#
# Verilog 는 "무엇을 계산하는가" 만 적는다. "얼마나 빨라야 하고 어느 핀에 붙는가" 는
# 여기 적는다. 문법은 Tcl 이다.
#
# 보드: 한백전자 FPGA Digital Logic Design 실습장비, XC7S75 (FGGA484) 모듈.
#       근거는 전부 `docs/hanback_kit.md` -- 킷 매뉴얼 CH2 94~99 쪽.
#
#   read_xdc rtl/constraints/kws_top.xdc

# =========================================================================== #
# [1] 타이밍
# =========================================================================== #

# --- 시스템 클럭: 50 MHz (MAIN_CLOCK, 클럭 선택 스위치 F) -------------------- #
# 킷의 Clock control block 이 0 Hz~50 MHz 를 16 단으로 내보내고 F 가 최고단이다.
# 성능이 아니라 여유로 고른 값이다: tb_top 이 기록한 클립당 비용이 2.87 M 사이클
# 이므로 50 MHz 에서 57.4 ms 이고, 클립 자체가 1.28 s(128 x 10 ms) 라 22배 남는다.
# 5 MHz 로 내려도 실시간이 성립한다 -- 타이밍이 안 닫히면 클럭을 낮추면 되지 설계를
# 고칠 일이 아니다.
#
# 이 주기가 kws_frame_ctrl 의 FRAME_CYCLES 를 정한다:
#     FRAME_CYCLES = clk_hz x 10 / 1000 = 50e6 x 0.01 = 500,000
create_clock -name sys_clk -period 20.000 [get_ports clk]

# B6 는 **MRCC 다** (`rtl/probe_part.tcl` 실측, 2026-09-08):
#
#     MAIN_CLOCK1    B6     bank 36   IO_L13P_T2_MRCC_36
#
# 그래서 여기 있던
#
#     set_property CLOCK_DEDICATED_ROUTE ANY [get_nets -quiet clk_IBUF]
#
# 는 지웠다. 「일반 I/O 로 밝혀지면」을 전제로 걸어둔 우회인데 그 전제가 깨졌다.
# 게다가 이 줄은 합성을 **깨뜨린다**: 제약이 엘라보레이션 단계에서 처리되는데
# 그때는 `clk_IBUF` 넷이 아직 없고, `-quiet` 는 get_nets 의 경고만 죽일 뿐
# 빈 결과를 받은 set_property 는 그대로 에러를 낸다.
#
#     ERROR: [Common 17-55] 'set_property' expects at least one object.
#
# MAIN_CLOCK 은 넷 다 MRCC 이므로(M8/M15/P15) 어느 것으로 옮겨도 이 우회는
# 필요 없다. docs/hanback_kit.md 3.3.

# --- 비동기 리셋 ------------------------------------------------------------ #
# rst_n 은 클럭과 무관하게 눌린다. 릴리스만 동기화되면 되므로 입력 경로는 뺀다.
set_false_path -from [get_ports rst_n]

# --- 비교기 16가닥 (kws_frame_ctrl 을 합성할 때만) --------------------------- #
# ICD 5: cmp 는 시스템 클럭과 무관한 비동기 신호이고 2단 플립플롭으로 받는다.
# 그 첫 단은 정의상 셋업/홀드를 만족할 수 없으므로 분석에서 빼야 한다. 안 빼면
# 도구가 존재하지 않는 경로를 맞추려다 실패하거나, 더 나쁘게는 맞추려고 로직을
# 비튼다.
#
# **`-quiet` 로는 부족하다** (2026-09-08 에 실측). `-quiet` 는 get_ports 가
# "못 찾았다" 고 내는 경고만 죽인다. 빈 목록을 받은 **set_false_path 자체가**
# 에러를 낸다:
#
#     ERROR: [Vivado 12-4739] set_false_path: No valid object(s) found for
#            '-from [get_ports -quiet {cmp[*]}]'.
#
# kws_top_synth 에는 cmp 가 없으므로(아래 [3] 절) 매번 여기서 멈춘다. 그래서
# 존재를 먼저 확인하고 건다. 이 파일은 read_xdc -unmanaged 로 읽히므로 if 를
# 쓸 수 있다 -- 평범한 XDC 였으면 if 가 거부됐을 것이다(rtl/build.tcl 참고).
if {[llength [get_ports -quiet {cmp[*]}]] > 0} {
    set_false_path -from [get_ports {cmp[*]}]
}

# 2단 동기화기 자체에는 ASYNC_REG 가 필요하다 -- 두 FF 를 같은 슬라이스에 묶어
# 메타스테이빌리티 해소 시간을 벌어준다. RTL 에 속성이 없으면 여기서 건다.
# 위와 같은 이유로 존재를 먼저 확인한다 (frame_ctrl 이 없으면 동기화기도 없다).
set _sync [get_cells -quiet -hier -filter {NAME =~ *sync_ff*}]
if {[llength $_sync] > 0} { set_property ASYNC_REG TRUE $_sync }
unset _sync

# =========================================================================== #
# [2] 핀 배치 — Expansion Port (J6)
# =========================================================================== #
#
# 이 절은 2026-09-07 에 미결에서 확정으로 바뀌었다. 킷 매뉴얼 99 쪽의 핀 구성표가
# EXT0~EXT45 <-> FPGA 볼을 전부 준다. 전체 표는 docs/hanback_kit.md 2.2.
#
# J6 = HEADER-25X2-POL (2x25, 키홈 박스 헤더):
#     핀 1,2   = VCC5V          <- AFE 전원을 여기서 받는다
#     핀 3~48  = EXT0~EXT45     <- 커넥터 핀 = EXT 번호 + 3
#     핀 49,50 = GND
# 모든 EXT 라인에 직렬 33 Ohm 이 들어가 있다(FPGA 입력이 고임피던스라 무해).
#
# cmp 를 연속 번호로 두는 이유: 리본 도선 번호와 채널 번호가 덧셈 하나로 묶여서
# 세기 쉽다. 채널이 섞이면 증상이 "정확도가 좀 낮다" 뿐이라 증상으로는 못 찾는다
# -- 배선 후 주파수 스윕 대각선 테스트(ICD 7.1)는 생략 불가다.

# EXT0 .. EXT45 의 볼 전부. 인덱스가 곧 EXT 번호다.
# (probe_part.tcl 로 46 개를 다 조회했다 -- 2026-09-08)
set EXT {
    Y21  AA22 AB21 AA21 AA20 Y20  Y19  AB20
    Y18  AB19 AB18 AA18 Y17  W17  AB17 AA17
    V16  U16  AA16 W16  T15  AB16 V15  U15
    AA15 W15  V14  T14  Y14  W14  AB14 AA14
    V13  T13  AA13 Y13  U12  AB13 W12  V12
    AB12 Y12  Y11  W11  AB11 AA11
}

# --- cmp 의 시작점 -- 동료 AFE 기판 배선에서 온다 (확정 2026-09-09) --------- #
# 동료가 J102(AFE 쪽 2x25) 를 이렇게 배선했다:
#
#     비교기 0 -> 커넥터 핀 33   ...   비교기 15 -> 커넥터 핀 48
#
# 1:1 스트레이트 리본이므로 킷 J6 도 같은 핀 번호이고, `EXT = 핀 - 3` 이니
#
#     cmp[k] -> EXT(30 + k)        k = 0..15   ->  EXT30 .. EXT45
#
# EXT45 가 확장 포트의 마지막 핀이다. 딱 끝까지 쓴다.
#
# **16 개가 전부 뱅크 13 이다** (probe_part.tcl). 종전 EXT0~15 는 뱅크 14 아홉 +
# 13 일곱으로 갈려 있었는데, 동료 배선이 우연히 한 도메인으로 떨어졌다.
# VCCO 가 양쪽 다 3.3 V 라 전기적 차이는 없지만 도메인이 하나면 더 안전하다.
#
# 이 값 하나만 바꾸면 배선이 바뀌어도 RTL 도 가중치도 .hex 도 그대로다.
# docs/ICD.md 가 존재하는 이유가 이것이다.
set CMP_EXT0 30

# --- IOSTANDARD: ✅ 실측으로 확정 (2026-09-09) ------------------------------ #
# vcco_probe 를 올리고 J6 에서 잰 값: **3.308 V**. 뱅크 13 과 14 가 같은 값이다.
# (재는 법과 판정표는 rtl/bringup/README.md 1, 원리는 docs/hanback_kit.md 4.4)
#
# 그러니 LVCMOS33 이 맞다 -- 종전에도 같은 값이었지만 그때는 "틀려도 무해한 쪽"
# 이라서 고른 잠정값이었고, 지금은 확정이다.
#
# 이 값이 아날로그 쪽에 뜻하는 것:
#
#   LVCMOS33 V_IH = 2.000 V
#   비교기 LPV7215 @1.8 V 최악 VOH = 1.63 V
#   -> **미달.** 레벨 변환기가 필요하다.
#
# 동료 기판이 SN74LXC8T245 두 개(VCCA 1.8 V / VCCB 3.3 V)로 그걸 한다.
# 그 A 측 슈미트 문턱 VT+ 는 1.8 V 에서 약 1.23 V 라 비교기 1.63 V 대비 여유
# +400 mV 다 (데이터시트 6.5 표를 1.65 V 행에서 보간). docs/hanback_kit.md 4.2.
set IOSTD LVCMOS33

# --- 배정 ------------------------------------------------------------------- #
# 포트가 없으면 조용히 건너뛴다. 최상위가 kws_top_synth 냐 (cmp 없음) 보드 래퍼냐
# (cmp 있음) 에 따라 존재하는 포트가 다르기 때문이다.
proc kws_pin {port ball iostd} {
    set p [get_ports -quiet $port]
    if {[llength $p] == 0} { return 0 }
    set_property PACKAGE_PIN $ball $p
    set_property IOSTANDARD  $iostd $p
    return 1
}

set n 0

# 비교기 16가닥 -> EXT30..EXT45 -> 커넥터 핀 33..48 (전부 뱅크 13)
for {set c 0} {$c < 16} {incr c} {
    incr n [kws_pin "cmp\[$c\]" [lindex $EXT [expr {$CMP_EXT0 + $c}]] $IOSTD]
}

# 제어 입력. 원래는 보드의 버튼/DIP 스위치가 맞는데 그 핀 구성표가 아직 없어서
# 확장 포트로 뺐다. 케이블을 안 물려도 안전한 값으로 읽히도록 내부 저항을 건다 --
# rst_n 은 풀업(리셋 해제), start 는 풀다운(시작 안 함).
#
# EXT16~23 은 cmp(EXT30~45)와 겹치지 않는다. 그리고 동료 AFE 기판은 그 핀들을
# 연결하지 않았으므로(신호는 33~48, 전원 1~2, 접지 49~50 뿐) 리본을 물려도
# 이쪽은 여전히 미연결이고, 위 내부 저항이 그대로 안전값을 만든다.
incr n [kws_pin rst_n [lindex $EXT 16] $IOSTD]
incr n [kws_pin start [lindex $EXT 17] $IOSTD]
set_property PULLUP   true [get_ports -quiet rst_n]
set_property PULLDOWN true [get_ports -quiet start]

# 결과 출력. LED/FND 핀 구성표가 오면 그쪽으로 옮기는 편이 훨씬 낫다 -- 분류
# 결과를 눈으로 보는 것이 첫 브링업에서 가장 값싼 관측 수단이다.
incr n [kws_pin class_valid [lindex $EXT 18] $IOSTD]
for {set b 0} {$b < 4} {incr b} {
    incr n [kws_pin "class_idx\[$b\]" [lindex $EXT [expr {19 + $b}]] $IOSTD]
}
incr n [kws_pin busy [lindex $EXT 23] $IOSTD]

# 클럭. MAIN_CLOCK1 = B6. 넷 중 어느 것이 실제로 우리 모듈에 연결되는지는 매뉴얼에
# 없다 -- M8 / M15 / P15 가 대안이다.
incr n [kws_pin clk B6 $IOSTD]

puts "== applied $n pin constraints (IOSTANDARD $IOSTD) =="

# =========================================================================== #
# [3] 아직 없는 것 — 보드 레벨 래퍼
# =========================================================================== #
#
# kws_frame_ctrl 과 kws_top 은 별도 모듈이고, 둘을 묶어 클럭/리셋을 붙이는 최상위가
# 아직 없다. 위 [2] 는 그 래퍼의 포트 이름을 미리 가정하고 쓴 것이라, 지금
# kws_top_synth 를 합성하면 cmp/class_* 는 없고 clk/rst_n/start 만 걸린다.
#
# 그래서 -impl(배치배선 + 비트스트림)은 래퍼가 생긴 뒤에 의미가 있다. P5-g.
#
# 래퍼를 쓸 때 같이 해야 하는 것: tb 에서 frame_ctrl 과 kws_top 을 **함께** 돌려
# 실시간 어서션을 실제로 밟아 보는 것. 지금까지 둘은 따로만 검증됐고(tb_frame_ctrl
# 은 FRAME_CYCLES=24 로 홀로, tb_top 은 프레임을 직접 먹임), "22배 여유" 는 두
# 숫자를 나눠서 얻은 값이지 실행으로 확인한 게 아니다.
