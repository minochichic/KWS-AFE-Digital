# 자체 검사 비트스트림을 올리고, VIO 로 두 번 돌려 결과를 읽는다 -- GUI 없이.
#
#   vivado -mode batch -source rtl/selftest/run_selftest.tcl -nolog -nojournal -notrace
#   vivado -mode batch -source rtl/selftest/run_selftest.tcl -nolog -nojournal -notrace -tclargs -runs 1 -noprog
#
# 리포 루트에서 돌린다. 기본 입력은 build.tcl -top kws_selftest_board -impl 의
# 산출물(out/synth/xc7s75fgga484-1/kws_selftest_board.{bit,ltx}).
#
# ---- 무엇을 보는가 ---------------------------------------------------------- //
#
#   match == total == CLIPS, any_fail == 0
#       칩이 export/golden.py 의 정수 경로(predictions_fixed.txt)를 **클립마다
#       정확히** 재현했다. 근사가 없는 비교라 하나라도 틀리면 버그다.
#
#   두 번 돌려 같은 값
#       결정성. 타이밍 위반이나 초기화 안 된 레지스터는 실행마다 다르게 틀린다.
#
# 정확도(82% 같은 수)는 여기서 안 나온다. 칩이 비교하는 상대는 정답 라벨이 아니라
# 파이썬의 예측이다 -- 둘이 전부 일치하면 칩의 정확도는 곧 파이썬 정수 경로의
# 정확도이고, 그 수는 labels.txt 와 predictions_fixed.txt 로 따로 센다.
#
# 이 스크립트는 AFE 핀을 구동하지 않는다(핀은 clk, rst_n 뿐). J6 에 뭐가 꽂혀
# 있어도 안전하다.

set ROOT [file normalize [file join [file dirname [info script]] ../..]]
cd $ROOT

set part   xc7s75fgga484-1
set top    kws_selftest_board
set runs   2
set prog   1
set tmo_s  180
# -dir <dir>: build.tcl -out 과 같은 값. 기본은 out/synth/<part>.
set dir    ""
for {set i 0} {$i < [llength $argv]} {incr i} {
    switch -- [lindex $argv $i] {
        -dir     { set dir   [lindex $argv [incr i]] }
        -part    { set part  [lindex $argv [incr i]] }
        -runs    { set runs  [lindex $argv [incr i]] }
        -timeout { set tmo_s [lindex $argv [incr i]] }
        -noprog  { set prog 0 }
        default { puts "unknown arg: [lindex $argv $i]"; exit 1 }
    }
}
if {$dir eq ""} { set dir out/synth/$part }
set bit $dir/$top.bit
set ltx $dir/$top.ltx
foreach f [list $bit $ltx] {
    if {![file exists $f]} {
        puts "ERROR: $f not found. Build it first:"
        puts "       vivado -mode batch -source rtl/build.tcl -notrace -tclargs \\"
        puts "           -tag bd_base -top $top -selftest out/selftest/bd_base -impl"
        exit 1
    }
}

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
current_hw_device $dev

set_property PROBES.FILE $ltx $dev
set_property FULL_PROBES.FILE $ltx $dev
if {$prog} {
    set_property PROGRAM.FILE $bit $dev
    program_hw_devices $dev
}
refresh_hw_device $dev
puts "== DONE = [get_property -quiet REGISTER.IR.BIT5_DONE $dev] =="

set vio [lindex [get_hw_vios -quiet -of_objects $dev] 0]
if {$vio eq ""} {
    puts "ERROR: no VIO found. Wrong bitstream, or .ltx does not match it."
    exit 1
}

# ltx 의 프로브 이름은 VIO 포트에 물린 **넷 이름**이다(kws_selftest_board.v).
# 계층 접두사가 붙을 수 있어서 끝부분으로 찾는다.
proc probe {vio name} {
    foreach p [get_hw_probes -of_objects $vio] {
        if {[regexp "(^|/)${name}(\\\[.*\\\])?\$" $p]} { return $p }
    }
    puts "ERROR: probe '$name' not in [get_hw_probes -of_objects $vio]"
    exit 1
}
set IN {done total match any_fail first_fail_idx first_fail_got first_fail_exp}
foreach n $IN {
    set P($n) [probe $vio $n]
    set_property INPUT_VALUE_RADIX UNSIGNED $P($n)
}
set P(go)  [probe $vio vio_go]
set P(rst) [probe $vio vio_rst]

proc drive {p v} {
    set_property OUTPUT_VALUE $v $p
    commit_hw_vio $p
}
proc sample {vio} {
    upvar P P IN IN
    refresh_hw_vio $vio
    set r {}
    foreach n $IN { dict set r $n [get_property INPUT_VALUE $P($n)] }
    return $r
}

set results {}
for {set k 1} {$k <= $runs} {incr k} {
    puts "\n== run $k / $runs =="
    # 매 실행 전 리셋: any_fail/first_fail_* 는 go 로 안 지워진다
    # (kws_selftest_board.v 머리말).
    drive $P(go)  0
    drive $P(rst) 1
    after 100
    drive $P(rst) 0
    after 100

    set t0 [clock milliseconds]
    drive $P(go) 1
    while 1 {
        after 2000
        set r [sample $vio]
        set el [expr {([clock milliseconds] - $t0) / 1000.0}]
        puts [format "   %5.1f s  total %4d  match %4d  any_fail %d" \
            $el [dict get $r total] [dict get $r match] [dict get $r any_fail]]
        if {[dict get $r done] == 1} break
        if {$el > $tmo_s} {
            puts "ERROR: not done after $tmo_s s -- harness stuck?"
            break
        }
    }
    drive $P(go) 0
    set el [expr {([clock milliseconds] - $t0) / 1000.0}]
    set tot [dict get $r total]
    if {$tot > 0} {
        puts [format "   %.1f s for %d clips = %.1f ms/clip (poll resolution 2 s)" \
            $el $tot [expr {$el * 1000.0 / $tot}]]
    }
    if {[dict get $r any_fail]} {
        puts "   FIRST MISMATCH: clip [dict get $r first_fail_idx]\
              got [dict get $r first_fail_got] want [dict get $r first_fail_exp]"
        puts "   (clip index is within the slice; add KWS_ST_BASE from paths.vh)"
    }
    lappend results $r
}

puts "\n== summary =="
set ok 1
foreach r $results {
    if {[dict get $r done] != 1 || [dict get $r any_fail] != 0 || \
        [dict get $r match] != [dict get $r total]} { set ok 0 }
}
if {[llength $results] > 1} {
    set first [lindex $results 0]
    foreach r [lrange $results 1 end] {
        if {$r ne $first} {
            puts "NOT DETERMINISTIC: runs differ"
            set ok 0
        }
    }
}
set r0 [lindex $results 0]
puts "total [dict get $r0 total]  match [dict get $r0 match]  any_fail [dict get $r0 any_fail]"
if {$ok} {
    puts "PASS chip reproduces predictions_fixed.txt on every clip, $runs run(s) identical"
} else {
    puts "FAIL see above"
}

close_hw_target
disconnect_hw_server
