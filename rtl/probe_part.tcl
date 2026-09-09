# 파트 조회 — Vivado 가 설치되면 가장 먼저 돌릴 것.
#
#   vivado -mode batch -source rtl/probe_part.tcl -nolog -nojournal -notrace
#
# 합성도 프로젝트도 없이 도구의 디바이스 DB 만 읽는다. 1 분이면 끝난다.
#
# ---- 출력이 왜 영어인가 (2026-09-08) --------------------------------------- #
#
# 이 파일은 UTF-8 인데 Vivado 의 Tcl 은 Windows 시스템 코드페이지(한국어 로캘에서
# cp949)로 읽는다. 그래서 `puts` 에 한글을 넣으면 콘솔에 깨져 나온다:
#
#     ===== 1. xc7s75* ?뙆?듃 紐⑸줉 =====
#
# 고칠 방법이 사실상 없다. 파일을 cp949 로 저장하면 git·에디터·리눅스가 깨지고,
# `encoding system utf-8` 은 이미 읽히는 중이라 늦으며, 윈도우의 "UTF-8 베타"
# 설정은 시스템 전역이라 대가가 크다. **콘솔에 나가는 문자열만 ASCII 로 둔다.**
#
# 주석은 한글 그대로다 -- 에디터에서 읽는 것이고, 콘솔에는 `-notrace` 로 안 나온다.
# `-notrace` 를 빼면 Vivado 가 소스 줄을 그대로 에코해서 주석까지 깨져 나온다.
#
# ---- 이것이 답해 주는 것 ---------------------------------------------------- #
#
#   1. xc7s75fgga484 의 속도 등급 목록.  매뉴얼은 -1 이라고 적었다. 확인만 한다.
#   2. Exp. Port 볼들이 어느 **뱅크**인가.  뱅크가 갈리면 VCCO 도 갈릴 수 있고,
#      그러면 IOSTANDARD 를 채널마다 다르게 줘야 한다.
#   3. B6 가 **클럭 가능 핀(MRCC/SRCC)** 인가.  아니면 kws_top.xdc 의
#      CLOCK_DEDICATED_ROUTE 우회가 필요하거나 M8/M15/P15 로 옮겨야 한다.
#
# 답하지 못하는 것: **VCCO 실제 전압.** 그건 보드 배선이지 칩 속성이 아니다.
# 예제 .xdc 의 IOSTANDARD 나 멀티미터로만 알 수 있다. docs/hanback_kit.md 4.
#
# ---- 2026-09-08 에 돌린 결과 (Vivado 2026.1) ------------------------------- #
#
#   1. -1 / -1IL / -1Q / -2 넷 다 있다. 매뉴얼의 -1 과 일치.
#   2. **갈린다.** EXT0~7·EXT9 -> 뱅크 14 (9개), 나머지 -> 뱅크 13 (37개).
#      "한 뱅크이길 기대한다" 가 빗나갔다. 경계는 패키지 18/19 행 사이다.
#      -> docs/hanback_kit.md 2.3, VCCO 를 뱅크마다 재야 한다.
#      다만 **뱅크 13 에 37 개**라 24 개(cmp 16 + 제어 8)를 통째로 옮길 수 있다.
#   3. B6 = IO_L13P_T2_MRCC_36. **MRCC 다.** MAIN_CLOCK 넷 다 MRCC 이고
#      USER_CLOCK 은 U18 만 MRCC 다. CLOCK_DEDICATED_ROUTE 우회 불필요.

set part xc7s75fgga484-1

puts "\n===== 1. xc7s75* part list ====="
foreach p [lsort [get_parts xc7s75*]] { puts "  $p" }

puts "\n===== 2. opening $part ====="
if {[llength [get_parts $part]] == 0} {
    puts "ERROR: $part is not in this installation."
    puts "       Check that Spartan-7 devices were selected during install."
    exit 1
}
link_design -part $part

# Exp. Port 46 가닥 전부 (docs/hanback_kit.md 2.2). 인덱스가 곧 EXT 번호다.
#
# 원래 EXT0~23 (우리가 쓰는 것) 만 조회했다. 전부로 넓힌 이유는 2026-09-08 에
# **뱅크가 13/14 로 갈린다**는 것이 밝혀졌기 때문이다. 두 뱅크의 VCCO 가 다르게
# 나오면 cmp 16 가닥을 한 뱅크 안으로 재배치해야 하는데, EXT0~23 범위의 뱅크 13
# 핀은 15 개뿐이라 하나가 모자란다. 46 개 중에서 골라야 여지가 생긴다.
set BALLS {
    Y21  AA22 AB21 AA21 AA20 Y20  Y19  AB20
    Y18  AB19 AB18 AA18 Y17  W17  AB17 AA17
    V16  U16  AA16 W16  T15  AB16 V15  U15
    AA15 W15  V14  T14  Y14  W14  AB14 AA14
    V13  T13  AA13 Y13  U12  AB13 W12  V12
    AB12 Y12  Y11  W11  AB11 AA11
}

puts "\n===== 3. Exp. Port balls: bank / pin function ====="
puts [format "%-6s %-6s %-6s %s" EXT BALL BANK PIN_FUNC]
set banks {}
for {set i 0} {$i < [llength $BALLS]} {incr i} {
    set b [lindex $BALLS $i]
    set pp [get_package_pins -quiet $b]
    if {[llength $pp] == 0} {
        puts [format "%-6s %-6s  ??    (ball not in this package)" EXT$i $b]
        continue
    }
    set bank [get_property BANK $pp]
    lappend banks $bank
    puts [format "%-6s %-6s %-6s %s" EXT$i $b $bank [get_property PIN_FUNC $pp]]
}
puts "\n  -> banks in use: [lsort -unique $banks]"
puts "     One bank: a single IOSTANDARD line is enough."
puts "     Several: VCCO may differ per bank -- measure each, constrain each."

# 뱅크별 개수 -- 재배치가 가능한지 바로 보인다. cmp 16 + 제어 8 = 24 가 필요하다.
puts "\n  EXT pins per bank (a bank with >= 24 can host all our signals):"
array set bcnt {}
foreach b $banks { incr bcnt($b) }
foreach b [lsort [array names bcnt]] {
    puts [format "    bank %-4s %2s pins" $b $bcnt($b)]
}

puts "\n===== 4. clock candidate pins ====="
puts [format "%-14s %-6s %-6s %s" SIGNAL BALL BANK PIN_FUNC]
foreach {sig ball} {MAIN_CLOCK1 B6 MAIN_CLOCK2 M8 MAIN_CLOCK3 M15 MAIN_CLOCK4 P15
                    USER_CLOCK1 D16 USER_CLOCK2 U18} {
    set pp [get_package_pins -quiet $ball]
    if {[llength $pp] == 0} { puts [format "%-14s %-6s  ??" $sig $ball]; continue }
    puts [format "%-14s %-6s %-6s %s" $sig $ball \
          [get_property BANK $pp] [get_property PIN_FUNC $pp]]
}
puts "\n  -> MRCC or SRCC in the name means a dedicated clock route is available."
puts "     Otherwise kws_top.xdc needs CLOCK_DEDICATED_ROUTE ANY, which only"
puts "     silences the warning -- prefer moving to a real clock pin instead."

puts "\n===== 5. part resources ====="
#
# 여기 원래 get_sites -filter 로 {SLICE_LUT6 RAMB36E1 DSP48E1 IOB} 를 셌고,
# 2026-09-08 에 돌려 보니 **넷 중 셋이 0** 이었다. 7-series 의 실제 SITE_TYPE
# 이름이 아니었기 때문인데, get_sites -filter 는 이름이 틀려도 에러 없이 빈
# 목록을 준다 -- "자원이 없다" 와 "이름을 잘못 물었다" 가 구별되지 않는다.
#
# get_sites 로 세는 것 자체도 틀린 접근이었다. 그건 **다이의 site 수**라
# 패키지가 실제로 내주는 양과 다르다: SLICEL+SLICEM 이 16,000 인데 이 파트의
# LUT 는 48,000 (= 슬라이스 12,000) 이고, DSP48E1 도 160 vs 140 이다.
#
# 파트 속성이 그 파트의 값을 직접 준다. 이름이 틀리면 빈 문자열이 오므로
# n/a 로 찍어 **조용한 0 이 아니라 눈에 보이는 실패**가 되게 한다.
set P [get_parts $part]
foreach {label prop} {
    "Speed grade"   SPEED
    "LUT elements"  LUT_ELEMENTS
    "Flip-flops"    FLIPFLOPS
    "Block RAMs"    BLOCK_RAMS
    "DSP slices"    DSP
    "Bonded IOBs"   AVAILABLE_IOBS
} {
    set v [get_property -quiet $prop $P]
    if {$v eq ""} { set v "n/a  <- property name may be wrong" }
    puts [format "  %-14s %s" $label $v]
}
puts "\n  (Manual's Default Part table: LUT 48,000 / FF 96,000 / DSP 140."
puts "   docs/hanback_kit.md 1. These should agree.)"

puts "\nDone. Record the results in docs/hanback_kit.md."
