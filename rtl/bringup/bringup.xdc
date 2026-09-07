# 브링업 제약 — vcco_probe / walk_probe 공용
#
# 핀은 킷 매뉴얼 99 쪽 핀 구성표에서 왔다(docs/hanback_kit.md 2.2). 본 설계의
# rtl/constraints/kws_top.xdc 와 **같은 핀**을 쓴다 -- 그래야 여기서 확인한 것이
# 본 설계에 그대로 옮겨진다.
#
#   EXT0..EXT15  ->  J6 핀 3..18  ->  ch00..ch15

set EXT {
    Y21  AA22 AB21 AA21 AA20 Y20  Y19  AB20
    Y18  AB19 AB18 AA18 Y17  W17  AB17 AA17
}

# --- IOSTANDARD: 모르는 채로 선언한다. 그래도 안전하다 ---------------------- #
# 이 비트스트림의 목적이 VCCO 를 재는 것이므로 VCCO 를 모르는 상태로 선언해야 한다.
# 괜찮은 이유: **IOSTANDARD 는 선언이지 설정이 아니다.** 출력 High 는 뱅크에 실제로
# 공급된 VCCO 까지 올라가고, 여기 뭘 적든 그 전압은 안 바뀐다. 틀리게 적었을 때
# 어긋나는 것은 도구의 드라이브·슬루 모델이지 전압이 아니다.
#
# DRIVE 를 최소로, SLEW 를 SLOW 로 두는 이유는 부하가 DMM(10 MΩ)뿐이라 전류가
# 필요 없고, 혹시 무언가 물려 있을 때 사고 전류를 줄여 주기 때문이다.
set IOSTD LVCMOS33

foreach c {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15} {
    set p [get_ports -quiet "ext\[$c\]"]
    if {[llength $p] == 0} { continue }
    set_property PACKAGE_PIN [lindex $EXT $c] $p
    set_property IOSTANDARD  $IOSTD $p
    set_property DRIVE       4      $p
    set_property SLEW        SLOW   $p
}

# --- 클럭: walk_probe 를 합성할 때만 --------------------------------------- #
# MAIN_CLOCK1 = B6 (docs/hanback_kit.md 3.1). 스위치 F 에서 50 MHz -> 20 ns.
# B6 가 클럭 가능 핀(MRCC/SRCC)인지 아직 확인 못 했다 -- rtl/probe_part.tcl 이
# 답한다. 아니면 M8 / M15 / P15 로 옮겨 다시 합성하면 된다.
if {[llength [get_ports -quiet clk]] > 0} {
    set_property PACKAGE_PIN B6      [get_ports clk]
    set_property IOSTANDARD  $IOSTD  [get_ports clk]
    create_clock -name sys_clk -period 20.000 [get_ports clk]
    # 일반 I/O 로 밝혀졌을 때 배치가 막히지 않도록. 경고를 지우는 것이지 좋은
    # 배치가 되는 건 아니므로, MRCC 핀이 따로 있으면 그쪽을 쓰는 편이 낫다.
    set_property CLOCK_DEDICATED_ROUTE ANY [get_nets -quiet clk_IBUF]
}

# --- 쓰지 않는 핀은 띄운다 -------------------------------------------------- #
# 기본값은 PULLDOWN 이라 나머지 322 개 I/O 가 전부 약하게 접지로 당겨진다. 이
# 보드는 그 핀들이 온보드 주변장치(LCD/7세그/SRAM/모터)에 물려 있으므로, 확인차
# 올리는 비트스트림이 그것들을 건드리지 않는 편이 안전하다.
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLNONE [current_design]
