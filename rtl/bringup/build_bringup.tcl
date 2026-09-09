# 브링업 비트스트림 — 본 설계와 완전히 분리해 둔다.
#
#   vivado -mode batch -source rtl/bringup/build_bringup.tcl
#   vivado -mode batch -source rtl/bringup/build_bringup.tcl -tclargs -top walk_probe
#   vivado -mode batch -source rtl/bringup/build_bringup.tcl -tclargs -part xc7s75fgga484-2
#
# 리포 루트에서 돌린다. 산출물은 out/bringup/<top>.bit.
#
# rtl/build.tcl 과 따로 두는 이유: 저쪽은 .hex 가중치를 싣고 태그를 고르고 ROM
# 경로를 다시 쓴다. 여기서 확인하려는 것은 그 어느 것도 아니라 **툴체인 · JTAG ·
# VCCO · 핀 배정** 뿐이고, 실패했을 때 원인이 섞이면 안 된다.

set ROOT [file normalize [file join [file dirname [info script]] ../..]]
cd $ROOT

set part xc7s75fgga484-1
set top  vcco_probe
for {set i 0} {$i < [llength $argv]} {incr i} {
    switch -- [lindex $argv $i] {
        -part { set part [lindex $argv [incr i]] }
        -top  { set top  [lindex $argv [incr i]] }
        default { puts "unknown arg: [lindex $argv $i]"; exit 1 }
    }
}
if {$top ni {vcco_probe walk_probe}} {
    puts "ERROR: -top must be vcco_probe or walk_probe (got: $top)"
    exit 1
}

set out out/bringup
file mkdir $out
# 콘솔로 나가는 문자열은 ASCII 로 둔다 -- Vivado 의 Tcl 이 이 UTF-8 파일을
# 시스템 코드페이지로 읽어서 한글이 깨진다. 이유는 rtl/probe_part.tcl 머리말.
puts "== bringup / part $part / top $top =="

create_project -in_memory -part $part
read_verilog rtl/bringup/$top.v

synth_design -top $top

# 제약은 read_xdc 가 아니라 source 로 넣는다 (2026.1 에서 실측, 2026-09-08).
#
# read_xdc 는 XDC 를 **제한된 Tcl 부분집합**으로 읽는다. foreach / if / proc 를
# 거부하고, 거부한 자리를 통째로 건너뛴다:
#
#   CRITICAL WARNING: [Designutils 20-1307] Command 'foreach' is not supported
#                     in the xdc constraint file. [bringup.xdc:24]
#
# bringup.xdc 는 핀 배정을 그 foreach **안에서** 하므로, 이러면 제약이 하나도
# 안 걸린 채 합성이 "성공" 한다. 잡아 주는 것은 write_bitstream 앞의 DRC 뿐이다
# (UCIO-1 / NSTD-1). 그게 없었으면 도구가 16 핀을 임의로 배정한 비트스트림이
# 나오고, 증상은 보드에서 "0 V" 하나였을 것이다.
#
# synth_design **뒤**인 이유: source 는 그 자리에서 즉시 실행되므로 get_ports 가
# 답하려면 디자인이 이미 있어야 한다. read_xdc 는 파일을 등록만 하고 합성 중에
# 파싱하기 때문에 앞에 놓을 수 있었다. 물리 제약(PACKAGE_PIN/IOSTANDARD)은
# place_design 전에만 있으면 되고, vcco_probe 는 클럭이 없어 타이밍으로 잃는
# 것도 없다. walk_probe 의 create_clock 도 place/route 전이면 충분하다.
source rtl/bringup/bringup.xdc

opt_design
place_design
route_design

report_utilization  -file $out/${top}_utilization.rpt
report_timing_summary -file $out/${top}_timing.rpt

# 핀이 하나라도 제약 없이 남으면 여기서 DRC 가 막는다. 그게 맞다 -- 미제약 핀은
# 도구가 임의로 배정하므로, 재려던 핀이 아닌 데로 신호가 나가고 증상은 "0 V" 뿐이다.
write_bitstream -force $out/$top.bit

# 실제로 그 핀에 갔는지 로그에 남긴다. 리포트를 사람이 안 열어봐도 보이도록.
puts "\n== pin assignment (compare against docs/hanback_kit.md 2.2) =="
foreach p [lsort -dictionary [get_ports]] {
    puts [format "   %-10s %-6s %s" $p \
          [get_property PACKAGE_PIN [get_ports $p]] \
          [get_property IOSTANDARD  [get_ports $p]]]
}
puts "\n== bitstream: $out/$top.bit =="
puts "== WARNING: UNPLUG THE AFE CABLE BEFORE LOADING THIS =="
puts "==          these 16 pins are driven as OUTPUTS; the comparators are"
puts "==          push-pull too, and the kit's 33 ohm series alone lets ~100 mA"
puts "==          flow. LPV7215 is a 580 nA micropower part. =="
