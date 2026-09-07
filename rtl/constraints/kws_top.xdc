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

# B6 가 클럭 가능 핀(MRCC/SRCC)인지 아직 확인 못 했다 -- UG475 의 볼-뱅크 표는
# 그림이라 텍스트가 없다. `rtl/probe_part.tcl` 이 Vivado 에서 한 줄로 답한다.
# 일반 I/O 로 밝혀지면 이 우회가 필요하고, 그때는 다른 MAIN_CLOCK 핀(M8/M15/P15)을
# 먼저 시도하는 편이 낫다.
set_property CLOCK_DEDICATED_ROUTE ANY [get_nets -quiet clk_IBUF]

# --- 비동기 리셋 ------------------------------------------------------------ #
# rst_n 은 클럭과 무관하게 눌린다. 릴리스만 동기화되면 되므로 입력 경로는 뺀다.
set_false_path -from [get_ports rst_n]

# --- 비교기 16가닥 (kws_frame_ctrl 을 합성할 때만) --------------------------- #
# ICD 5: cmp 는 시스템 클럭과 무관한 비동기 신호이고 2단 플립플롭으로 받는다.
# 그 첫 단은 정의상 셋업/홀드를 만족할 수 없으므로 분석에서 빼야 한다. 안 빼면
# 도구가 존재하지 않는 경로를 맞추려다 실패하거나, 더 나쁘게는 맞추려고 로직을
# 비튼다.
#
# quiet 인 이유: kws_top 만 합성하면 이 포트가 없다. 그때 조용히 넘어가야 한다.
set_false_path -from [get_ports -quiet {cmp[*]}]

# 2단 동기화기 자체에는 ASYNC_REG 가 필요하다 -- 두 FF 를 같은 슬라이스에 묶어
# 메타스테이빌리티 해소 시간을 벌어준다. RTL 에 속성이 없으면 여기서 건다.
set_property ASYNC_REG TRUE [get_cells -quiet -hier -filter {NAME =~ *sync_ff*}]

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
# cmp 를 EXT0 부터 연속으로 두는 이유: 리본 도선 번호와 채널 번호가
# `핀 = ch + 3` 하나로 묶여서 빨간 줄(1번 도선)부터 세기 쉽다. 채널이 섞이면
# 증상이 "정확도가 좀 낮다" 뿐이라 증상으로는 못 찾는다 -- 배선 후 주파수 스윕
# 대각선 테스트(ICD 7.1)는 생략 불가다.

# EXT0 .. EXT23 의 볼. 인덱스가 곧 EXT 번호다.
set EXT {
    Y21  AA22 AB21 AA21 AA20 Y20  Y19  AB20
    Y18  AB19 AB18 AA18 Y17  W17  AB17 AA17
    V16  U16  AA16 W16  T15  AB16 V15  U15
}

# --- IOSTANDARD: ⬜ 아직 모른다 --------------------------------------------- #
# Spartan-7 은 HR 뱅크만 있고 1.2~3.3 V 를 지원하므로(DS180) 칩 쪽 제약은 없다.
# 킷이 이 뱅크의 VCCO 를 몇 V 로 배선했는지가 전부이고, 매뉴얼 94~99 쪽에는 없다.
#
#   VCCO 1.8 V -> LVCMOS18, V_IH 1.17 V  : 비교기 최악 VOH 1.63 V 대비 +460 mV
#   VCCO 2.5 V -> LVCMOS25, V_IH 1.700 V : -70 mV                   ⚠️
#   VCCO 3.3 V -> LVCMOS33, V_IH 2.000 V : 미달 -> 변환기 또는 비교기를 3.3 V 로
#
# 온보드 주변장치가 3.3/5 V 계열이고 헤더로 5 V 를 내보내는 보드라 **3.3 V 일
# 가능성이 높다.** 예제 .xdc 한 줄이나 멀티미터 한 번이면 끝난다.
#
# 잘못 선언하면 조용히 위험하다: 3.3 V 로 배선된 뱅크에 LVCMOS18 을 선언해도
# VCCO 는 안 바뀌고(공급 핀이지 설정이 아니다) 출력 드라이브 예측만 틀어진다.
# 그래서 **확정 전까지 여기 값을 바꾸지 않는다.** 지금 값은 "가장 그럴듯한 쪽" 이
# 아니라 "틀려도 입력만 읽는 우리 설계에서 가장 무해한 쪽" 이다.
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

# 비교기 16가닥 -> EXT0..EXT15 -> 커넥터 핀 3..18
for {set c 0} {$c < 16} {incr c} {
    incr n [kws_pin "cmp\[$c\]" [lindex $EXT $c] $IOSTD]
}

# 제어 입력. 원래는 보드의 버튼/DIP 스위치가 맞는데 그 핀 구성표가 아직 없어서
# 확장 포트로 뺐다. 케이블을 안 물려도 안전한 값으로 읽히도록 내부 저항을 건다 --
# rst_n 은 풀업(리셋 해제), start 는 풀다운(시작 안 함).
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

puts "== 핀 제약 $n 개 적용 (IOSTANDARD $IOSTD) =="

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
