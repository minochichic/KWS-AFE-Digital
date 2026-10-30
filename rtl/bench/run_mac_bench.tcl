# MAC microbenchmark -- synthesise, place and route every bench_mac KIND on the
# kit's part and tabulate what each costs (docs/conv2_int8_compare.md, option A).
#
#   vivado -mode batch -source rtl/bench/run_mac_bench.tcl -notrace
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
    set dyn   [grab {\|\s*Dynamic \(W\)\s*\|\s*([0-9.]+)} $p]
    set tot   [grab {\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $p]
    set wns   [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
    lappend rows [list $k $name $tpc $lut $ff $dsp $wns $dyn $tot]
    close_design
}

# ---- subtract the stimulus (KIND 5) and tabulate ------------------------------ #
foreach r $rows { if {[lindex $r 0] == 5} { set base $r } }
set csv [open $out/mac_bench.csv w]
puts $csv "kind,name,terms_per_cycle,lut,ff,dsp,wns_ns,dyn_w,total_w,lut_net,ff_net,dyn_net_mw,pj_per_term"
puts ""
puts [format "%-12s %5s %7s %7s %4s %7s %9s %9s %10s" \
      name t/cyc LUT FF DSP WNS dyn_mW net_mW pJ/term]
foreach r $rows {
    lassign $r k name tpc lut ff dsp wns dyn tot
    set ln [expr {$lut - [lindex $base 3]}]
    set fn [expr {$ff  - [lindex $base 4]}]
    set dn [expr {($dyn - [lindex $base 7]) * 1000.0}]
    # the engine runs 64 of every 65 cycles (one start cycle per vector)
    set pj [expr {$tpc > 0 ? $dn * 1e-3 / ($clk_hz * $tpc * 64.0 / 65.0) * 1e12 : 0}]
    puts $csv "$k,$name,$tpc,$lut,$ff,$dsp,$wns,$dyn,$tot,$ln,$fn,[format %.2f $dn],[format %.3f $pj]"
    puts [format "%-12s %5d %7d %7d %4s %7.2f %9.1f %9.1f %10.3f" \
          $name $tpc $ln $fn $dsp $wns [expr {$dyn * 1000.0}] $dn $pj]
}
close $csv
puts "\nLUT / FF / net_mW are after subtracting KIND 5 (stimulus only)."
puts "pJ/term = engine dynamic power / terms per second at 50 MHz. Vectorless."
puts "written $out/mac_bench.csv"
