# 파트 조회 — Vivado 가 설치되면 가장 먼저 돌릴 것.
#
#   vivado -mode batch -source rtl/probe_part.tcl -nolog -nojournal
#
# 합성도 프로젝트도 없이 도구의 디바이스 DB 만 읽는다. 몇 초면 끝난다.
#
# 이것이 답해 주는 것 (전부 지금 문서에 "미확인" 으로 남아 있는 항목이다):
#
#   1. xc7s75fgga484 의 속도 등급 목록.  매뉴얼은 -1 이라고 적었다. 확인만 한다.
#   2. Exp. Port 볼들이 어느 **뱅크**인가.  뱅크가 갈리면 VCCO 도 갈릴 수 있고,
#      그러면 IOSTANDARD 를 채널마다 다르게 줘야 한다. (한 뱅크이길 기대한다.)
#   3. B6 가 **클럭 가능 핀(MRCC/SRCC)** 인가.  아니면 kws_top.xdc 의
#      CLOCK_DEDICATED_ROUTE 우회가 필요하거나 M8/M15/P15 로 옮겨야 한다.
#
# 답하지 못하는 것: **VCCO 실제 전압.** 그건 보드 배선이지 칩 속성이 아니다.
# 예제 .xdc 의 IOSTANDARD 나 멀티미터로만 알 수 있다. docs/hanback_kit.md 4.

set part xc7s75fgga484-1

puts "\n===== 1. xc7s75* 파트 목록 ====="
foreach p [lsort [get_parts xc7s75*]] { puts "  $p" }

puts "\n===== 2. $part 를 여는 중 ====="
if {[llength [get_parts $part]] == 0} {
    puts "ERROR: $part 가 설치본에 없다."
    puts "       Vivado 설치 때 Spartan-7 디바이스를 체크했는지 확인할 것."
    exit 1
}
link_design -part $part

# Exp. Port 에서 우리가 쓰는 볼 (docs/hanback_kit.md 6). EXT 번호 순.
set BALLS {
    Y21  AA22 AB21 AA21 AA20 Y20  Y19  AB20
    Y18  AB19 AB18 AA18 Y17  W17  AB17 AA17
    V16  U16  AA16 W16  T15  AB16 V15  U15
}

puts "\n===== 3. Exp. Port 볼의 뱅크 / 이름 ====="
puts [format "%-4s %-6s %-6s %s" EXT 볼 뱅크 이름]
set banks {}
for {set i 0} {$i < [llength $BALLS]} {incr i} {
    set b [lindex $BALLS $i]
    set pp [get_package_pins -quiet $b]
    if {[llength $pp] == 0} { puts [format "%-4s %-6s  ??  (패키지에 없는 볼)" EXT$i $b]; continue }
    set bank [get_property BANK $pp]
    set nm   [get_property PIN_FUNC $pp]
    lappend banks $bank
    puts [format "%-4s %-6s %-6s %s" EXT$i $b $bank $nm]
}
puts "\n  -> 뱅크 집합: [lsort -unique $banks]"
puts "     한 개면 IOSTANDARD 한 줄로 끝나고, 여러 개면 뱅크마다 나눠 줘야 한다."

puts "\n===== 4. 클럭 후보 핀 ====="
puts [format "%-14s %-6s %-6s %s" 신호 볼 뱅크 이름]
foreach {sig ball} {MAIN_CLOCK1 B6 MAIN_CLOCK2 M8 MAIN_CLOCK3 M15 MAIN_CLOCK4 P15
                    USER_CLOCK1 D16 USER_CLOCK2 U18} {
    set pp [get_package_pins -quiet $ball]
    if {[llength $pp] == 0} { puts [format "%-14s %-6s  ??" $sig $ball]; continue }
    puts [format "%-14s %-6s %-6s %s" $sig $ball \
          [get_property BANK $pp] [get_property PIN_FUNC $pp]]
}
puts "\n  -> 이름에 MRCC 나 SRCC 가 있으면 클럭 전용 경로를 쓸 수 있다."
puts "     없으면 kws_top.xdc 의 CLOCK_DEDICATED_ROUTE ANY 가 필요하고,"
puts "     그건 경고를 지우는 것이지 좋은 배치가 되는 건 아니다 -- 다른 핀을 먼저 본다."

puts "\n===== 5. 자원 ====="
foreach t {SLICE_LUT6 RAMB36E1 DSP48E1 IOB} {
    puts [format "  %-12s %s" $t [llength [get_sites -quiet -filter "SITE_TYPE == $t"]]]
}

puts "\n완료. 결과를 docs/hanback_kit.md 에 반영할 것."
