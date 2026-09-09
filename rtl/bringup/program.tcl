# 비트스트림을 보드에 올린다 -- GUI 없이.
#
#   vivado -mode batch -source rtl/bringup/program.tcl -nolog -nojournal -notrace
#   vivado -mode batch -source rtl/bringup/program.tcl -nolog -nojournal -notrace -tclargs -bit out/bringup/walk_probe.bit
#
# 리포 루트에서 돌린다. 기본 비트스트림은 out/bringup/vcco_probe.bit.
#
# ⚠️ 올리기 전에 확인할 것 -- 이 스크립트는 검사해 줄 수 없다:
#
#   J6 에 아무것도 안 꽂혀 있는가?
#
# vcco_probe / walk_probe 는 EXT0~15 를 **출력으로 구동**한다. AFE 가 물려 있으면
# 비교기 출력과 맞물려 출력끼리 싸운다. 킷의 직렬 33 옴만으로는 100 mA 가까이
# 흐르고 LPV7215 는 580 nA 마이크로파워 부품이라 못 견딘다.
# docs/hanback_kit.md 4.5 의 마지막 경고.
#
# 지금은 AFE 기판이 아직 없으므로 해당 없지만, 생긴 뒤에는 매번 확인해야 한다.
#
# 프로그래밍은 휘발성이다 -- 전원을 내리면 사라진다. Flash 에 굽지 않는다:
# 브링업은 몇 분 쓰고 버리는 것이고, Flash 는 되돌리는 데 더 손이 간다.

set ROOT [file normalize [file join [file dirname [info script]] ../..]]
cd $ROOT

set bit out/bringup/vcco_probe.bit
for {set i 0} {$i < [llength $argv]} {incr i} {
    switch -- [lindex $argv $i] {
        -bit { set bit [lindex $argv [incr i]] }
        default { puts "unknown arg: [lindex $argv $i]"; exit 1 }
    }
}

if {![file exists $bit]} {
    puts "ERROR: $bit not found. Build it first:"
    puts "       vivado -mode batch -source rtl/bringup/build_bringup.tcl"
    exit 1
}
puts "== bitstream: $bit =="

open_hw_manager
connect_hw_server -allow_non_jtag

set targets [get_hw_targets -quiet]
if {[llength $targets] == 0} {
    puts "ERROR: no JTAG target. Cable drivers / power / ribbon."
    exit 1
}
open_hw_target [lindex $targets 0]

set dev [lindex [get_hw_devices -quiet] 0]
if {$dev eq ""} { puts "ERROR: no device on the chain."; exit 1 }
puts "== device: $dev  (part [get_property -quiet PART $dev]) =="

current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
refresh_hw_device $dev

# DONE 이 섰는지가 유일하게 중요한 확인이다. 안 서면 전압을 재봐야 소용없다.
set done [get_property -quiet REGISTER.IR.BIT5_DONE $dev]
puts "\n== DONE = $done =="
if {$done ne "1"} {
    puts "   DONE is not 1. The bitstream did not take. Do NOT measure yet --"
    puts "   the pins are still inputs and any reading is meaningless."
} else {
    puts "   Configured. The 16 EXT pins are now driven HIGH."
    puts ""
    puts "   Measure, with the DMM in VOLTAGE mode, ground on J6 pin 49 or 50:"
    puts "     J6 pin  3  = EXT0  -> bank 14 VCCO"
    puts "     J6 pin 11  = EXT8  -> bank 13 VCCO"
    puts "   The two banks may legitimately differ. rtl/bringup/README.md 1."
}

close_hw_target
disconnect_hw_server
