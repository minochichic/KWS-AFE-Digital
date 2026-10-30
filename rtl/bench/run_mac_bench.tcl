# MAC microbenchmark -- synthesise, place and route every bench_mac KIND on the
# kit's part and tabulate what each costs (docs/conv2_int8_compare.md, option A).
#
#   vivado -mode batch -source rtl/bench/run_mac_bench.tcl -notrace
#   vivado -mode batch -source rtl/bench/run_mac_bench.tcl -notrace -tclargs 6 7
#                                   (only those KINDs; 5 is always added)
#   vivado -mode batch -source rtl/bench/run_mac_bench.tcl -notrace -tclargs 8 9 10 11 12
#                                   (engine + operand memory, ~18 min)
#
# Non-project, like rtl/build.tcl. About 3 minutes per KIND on the T470.
# Writes out/bench/mac_bench.csv and per-KIND reports under out/bench/.
#
# POWER IS VECTORLESS (report_power's default switching), with the stimulus
# generated inside the design. Good for comparing KINDs against each other;
# the absolute numbers are not a measurement. KIND 5 (stimulus only) is
# subtracted so the table shows the engine alone.

set part  xc7s75fgga484-1
set clk_hz 50e6
set out   out/bench
file mkdir $out

#        KIND name         terms/cycle
set kinds {
    0 bin32        32
    1 int8x32      32
    2 int8x1        1
    3 int8pm1x32   32
    4 int8x32_lut  32
    5 stim          0
    6 int8x32_dsp  32
    7 bin32x8     256
    8 bin32x8_mem 256
    9 int8x32_dsp_mem  32
   10 int8x32_dsp_bram 32
   11 int8pm1x32_mem   32
   12 bin32_mem         32
}
if {[llength $argv] > 0} {
    set want [lsort -unique [concat $argv 5]]
    set sel {}
    foreach {k n t} $kinds { if {[lsearch -exact $want $k] >= 0} { lappend sel $k $n $t } }
    set kinds $sel
}

read_verilog [list rtl/kws_bin_mac.v rtl/bench/bench_mac.v]
read_xdc rtl/bench/bench_mac.xdc

proc grab {re text} {
    if {[regexp $re $text -> v]} { return $v }
    return "NA"
}

set rows {}
foreach {k name tpc} $kinds {
    puts "==== KIND $k ($name) ===="
    synth_design -top bench_mac -part $part -generic KIND=$k -flatten_hierarchy rebuilt
    opt_design
    place_design
    route_design

    set u [report_utilization -return_string]
    set p [report_power -return_string]
    report_utilization -file $out/${name}_util.rpt
    report_power       -file $out/${name}_power.rpt
    report_timing_summary -file $out/${name}_timing.rpt

    set lut   [grab {\|\s*Slice LUTs\*?\s*\|\s*([0-9]+)} $u]
    set ff    [grab {\|\s*Slice Registers\s*\|\s*([0-9]+)} $u]
    set dsp   [grab {\|\s*DSPs\s*\|\s*([0-9]+)} $u]
    set bram  [grab {\|\s*Block RAM Tile\s*\|\s*([0-9.]+)} $u]
    set dyn   [grab {\|\s*Dynamic \(W\)\s*\|\s*([0-9.]+)} $p]
    set tot   [grab {\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $p]
    set wns   [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
    lappend rows [list $k $name $tpc $lut $ff $dsp $wns $dyn $tot $bram]
    close_design
}

# ---- subtract the stimulus (KIND 5) and tabulate ------------------------------ #
foreach r $rows { if {[lindex $r 0] == 5} { set base $r } }
set csv_name [expr {[llength $argv] > 0 ? "mac_bench_[join $argv _].csv" : "mac_bench.csv"}]
set csv [open $out/$csv_name w]
puts $csv "kind,name,terms_per_cycle,lut,ff,dsp,bram,wns_ns,dyn_w,total_w,lut_net,ff_net,dyn_net_mw,pj_per_term"
puts ""
puts [format "%-16s %5s %7s %7s %4s %5s %7s %9s %9s %10s" \
      name t/cyc LUT FF DSP BRAM WNS dyn_mW net_mW pJ/term]
foreach r $rows {
    lassign $r k name tpc lut ff dsp wns dyn tot bram
    set ln [expr {$lut - [lindex $base 3]}]
    set fn [expr {$ff  - [lindex $base 4]}]
    set dn [expr {($dyn - [lindex $base 7]) * 1000.0}]
    # the engine runs 64 of every 65 cycles (one start cycle per vector)
    set pj [expr {$tpc > 0 ? $dn * 1e-3 / ($clk_hz * $tpc * 64.0 / 65.0) * 1e12 : 0}]
    puts $csv "$k,$name,$tpc,$lut,$ff,$dsp,$bram,$wns,$dyn,$tot,$ln,$fn,[format %.2f $dn],[format %.3f $pj]"
    puts [format "%-16s %5d %7d %7d %4s %5s %7.2f %9.1f %9.1f %10.3f" \
          $name $tpc $ln $fn $dsp $bram $wns [expr {$dyn * 1000.0}] $dn $pj]
    # The memory kinds' ROMs are constants from an initial loop. If synthesis
    # ever failed to evaluate it, the ROM would be all zero and fold away, and
    # the "memory" row would cost nothing. Expected: 256x32 ROM ~ 128 LUT per
    # copy, 256x256 ~ 1,024 LUT, or 4 RAMB36 in block RAM.
    set floor [dict create 8 1500 9 1500 11 1500 12 250]
    if {[dict exists $floor $k] && $ln < [dict get $floor $k]} {
        puts "  SUSPECT: $name has $ln LUT (< [dict get $floor $k]) -- ROM folded away?"
    }
    if {$k == 10 && ($bram eq "NA" || $bram < 4)} {
        puts "  SUSPECT: $name has $bram block RAM tiles (< 4) -- ROM not in BRAM?"
    }
}
close $csv
puts "\nLUT / FF / net_mW are after subtracting KIND 5 (stimulus only)."
puts "pJ/term = engine dynamic power / terms per second at 50 MHz. Vectorless."
puts "written $out/$csv_name"
