# Program the streaming self-test bitstream and read it over VIO -- no GUI.
#
#   vivado -mode batch -source rtl/selftest/run_stream_selftest.tcl -nolog -nojournal -notrace -tclargs -dir out/build/p75_stream
#   ... -tclargs -dir out/build/p75_stream -runs 1 -noprog
#
# Pass: every run ends with match == total == CASES, any_fail 0, overrun 0, and
# the hit / wrong / quiet_false counters equal subset_summary in
# stream_manifest.json. One case is 300 frames at 10 ms = 3 s, so 240 cases take
# about 12 minutes per run.
#
# Check the base-board clock display shows 50 MHz first (docs/hanback_kit.md 3.2):
# at 0 Hz the debug hub is not detected.

set ROOT [file normalize [file join [file dirname [info script]] ../..]]
cd $ROOT

set dir   ""
set part  xc7s75fgga484-1
set top   kws_stream_selftest_board
set runs  2
set prog  1
set poll  10
set tmo_s 1800
for {set i 0} {$i < [llength $argv]} {incr i} {
    switch -- [lindex $argv $i] {
        -dir     { set dir   [lindex $argv [incr i]] }
        -runs    { set runs  [lindex $argv [incr i]] }
        -timeout { set tmo_s [lindex $argv [incr i]] }
        -poll    { set poll  [lindex $argv [incr i]] }
        -noprog  { set prog 0 }
        default  { puts "unknown arg: [lindex $argv $i]"; exit 1 }
    }
}
if {$dir eq ""} { set dir out/synth/$part }
set bit $dir/$top.bit
set ltx $dir/$top.ltx
foreach f [list $bit $ltx] {
    if {![file exists $f]} { puts "ERROR: $f not found (build.tcl -top $top -stream-selftest ... -out $dir -impl)"; exit 1 }
}

open_hw_manager
connect_hw_server -allow_non_jtag
set targets [get_hw_targets -quiet]
if {[llength $targets] == 0} { puts "ERROR: no JTAG target"; exit 1 }
open_hw_target [lindex $targets 0]
set dev [lindex [get_hw_devices -quiet] 0]
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
if {$vio eq ""} { puts "ERROR: no VIO found (clock switch at 50 MHz? matching .ltx?)"; exit 1 }

proc probe {vio name} {
    foreach p [get_hw_probes -of_objects $vio] {
        if {[regexp "(^|/)${name}(\\\[.*\\\])?\$" $p]} { return $p }
    }
    puts "ERROR: probe '$name' not in [get_hw_probes -of_objects $vio]"
    exit 1
}
set DEC {done total match any_fail first_fail_case hit wrong quiet_false outside window_now overrun_count}
set HEX {first_fail_got first_fail_exp}
foreach n $DEC { set P($n) [probe $vio $n]; set_property INPUT_VALUE_RADIX UNSIGNED $P($n) }
foreach n $HEX { set P($n) [probe $vio $n]; set_property INPUT_VALUE_RADIX HEX $P($n) }
set P(go)  [probe $vio vio_go]
set P(rst) [probe $vio vio_rst]

proc drive {p v} { set_property OUTPUT_VALUE $v $p; commit_hw_vio $p }
proc sample {vio} {
    upvar P P DEC DEC HEX HEX
    refresh_hw_vio $vio
    set r {}
    foreach n [concat $DEC $HEX] { dict set r $n [get_property INPUT_VALUE $P($n)] }
    return $r
}

set results {}
for {set k 1} {$k <= $runs} {incr k} {
    puts "\n== run $k / $runs =="
    drive $P(go) 0
    drive $P(rst) 1
    after 100
    drive $P(rst) 0
    after 100
    set t0 [clock milliseconds]
    drive $P(go) 1
    while 1 {
        after [expr {$poll * 1000}]
        set r [sample $vio]
        set el [expr {([clock milliseconds] - $t0) / 1000.0}]
        puts [format "  %6.0f s  case %3d  win %2d  match %3d  hit %3d  wrong %2d  quiet_false %2d  outside %d  overrun %d  any_fail %d" \
            $el [dict get $r total] [dict get $r window_now] [dict get $r match] [dict get $r hit] \
            [dict get $r wrong] [dict get $r quiet_false] [dict get $r outside] \
            [dict get $r overrun_count] [dict get $r any_fail]]
        if {[dict get $r done] == 1} break
        if {$el > $tmo_s} { puts "ERROR: not done after $tmo_s s"; break }
    }
    drive $P(go) 0
    if {[dict get $r any_fail]} {
        puts "  FIRST MISMATCH: case [dict get $r first_fail_case]  got [dict get $r first_fail_got]  exp [dict get $r first_fail_exp]"
        puts "  (word = crc16 | target | det1 {v,win,idx} | det0; compare with stream_cases.csv of the slice)"
    }
    lappend results $r
}

puts "\n== summary =="
set ok 1
foreach r $results {
    if {[dict get $r done] != 1 || [dict get $r any_fail] != 0 || [dict get $r overrun_count] != 0 \
            || [dict get $r match] != [dict get $r total]} { set ok 0 }
}
set first [lindex $results 0]
foreach r [lrange $results 1 end] { if {$r ne $first} { puts "NOT DETERMINISTIC: runs differ"; set ok 0 } }
puts "total [dict get $first total]  match [dict get $first match]  hit [dict get $first hit]  wrong [dict get $first wrong]  quiet_false [dict get $first quiet_false]  outside [dict get $first outside]  overrun [dict get $first overrun_count]"
if {$ok} {
    puts "PASS chip reproduces the integer streaming reference on every case, $runs run(s) identical"
} else {
    puts "FAIL see above"
}
close_hw_target
disconnect_hw_server
