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
    puts "ERROR: -top 은 vcco_probe 또는 walk_probe 여야 한다 (받은 값: $top)"
    exit 1
}

set out out/bringup
file mkdir $out
puts "== 브링업 / part $part / top $top =="

create_project -in_memory -part $part
read_verilog rtl/bringup/$top.v
read_xdc     rtl/bringup/bringup.xdc

synth_design -top $top
opt_design
place_design
route_design

report_utilization  -file $out/${top}_utilization.rpt
report_timing_summary -file $out/${top}_timing.rpt

# 핀이 하나라도 제약 없이 남으면 여기서 DRC 가 막는다. 그게 맞다 -- 미제약 핀은
# 도구가 임의로 배정하므로, 재려던 핀이 아닌 데로 신호가 나가고 증상은 "0 V" 뿐이다.
write_bitstream -force $out/$top.bit

# 실제로 그 핀에 갔는지 로그에 남긴다. 리포트를 사람이 안 열어봐도 보이도록.
puts "\n== 핀 배정 확인 =="
foreach p [lsort -dictionary [get_ports]] {
    puts [format "   %-10s %-6s %s" $p \
          [get_property PACKAGE_PIN [get_ports $p]] \
          [get_property IOSTANDARD  [get_ports $p]]]
}
puts "\n== 비트스트림: $out/$top.bit =="
puts "== ⚠️ 올리기 전에 AFE 케이블을 뽑을 것 -- 이 핀들은 출력으로 구동된다 =="
