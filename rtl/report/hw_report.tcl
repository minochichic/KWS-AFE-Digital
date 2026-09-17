# Hardware numbers and pictures for the interim report, from a routed checkpoint.
#
#   vivado -mode batch -source rtl/report/hw_report.tcl -notrace -tclargs -dir out/build/bd_base_bal
#   vivado -mode gui   -source rtl/report/hw_report.tcl -notrace -tclargs -dir out/build/bd_base_bal
#
# Batch mode writes every report. The PICTURES (schematic, placed device) need a
# window to draw into, so batch mode skips them with a note and gui mode writes
# them too. Run gui mode once; it closes itself when done unless -keep.
#
# Output: <dir>/report/  -- nothing here is committed, it is regenerated.

set ROOT [file normalize [file join [file dirname [info script]] ../..]]
cd $ROOT

set dir  ""
set keep 0
for {set i 0} {$i < [llength $argv]} {incr i} {
    switch -- [lindex $argv $i] {
        -dir  { set dir [lindex $argv [incr i]] }
        -keep { set keep 1 }
        default { puts "unknown arg: [lindex $argv $i]"; exit 1 }
    }
}
if {$dir eq "" || ![file exists $dir/post_route.dcp]} {
    puts "ERROR: -dir must hold post_route.dcp (a build.tcl -impl output)"
    exit 1
}
set rep $dir/report
file mkdir $rep

open_checkpoint $dir/post_route.dcp

# ---- tables ---------------------------------------------------------------- #
# Hierarchical to depth 3 is where the design reads as blocks: network vs
# harness vs debug, and inside the network one row per layer.
report_utilization -hierarchical -hierarchical_depth 4 -file $rep/util_hier.rpt
report_utilization -file $rep/util_summary.rpt
report_timing_summary -max_paths 10 -file $rep/timing_summary.rpt
report_timing -max_paths 20 -nworst 1 -unique_pins -file $rep/timing_top20.rpt
# The worst path per major block: tells whether the critical path is a design
# property (the MAC) or an accident of placement.
foreach blk {u_top/u_net/u_top/u_c1 u_top/u_net/u_top/u_b1 u_top/u_net/u_top/u_b2
             u_top/u_net/u_top/u_b3 u_top/u_net/u_top/u_c2 u_top/u_net/u_top/u_tail
             u_top/u_st} {
    set cells [get_cells -quiet -hier -filter "NAME =~ $blk/*"]
    if {[llength $cells]} {
        set tag [string map {/ _} $blk]
        report_timing -to $cells -max_paths 1 -file $rep/timing_to_$tag.rpt
    }
}
report_power -hierarchical_depth 4 -file $rep/power_hier.rpt
report_clock_utilization -file $rep/clock_util.rpt
report_drc -file $rep/drc.rpt
report_methodology -file $rep/methodology.rpt
report_route_status -file $rep/route_status.rpt

# Slack histogram data: one line per endpoint, for a figure drawn elsewhere.
set fh [open $rep/setup_slack_endpoints.csv w]
puts $fh "endpoint,slack_ns"
foreach p [get_timing_paths -max_paths 20000 -nworst 1 -setup] {
    puts $fh "[get_property ENDPOINT_PIN $p],[get_property SLACK $p]"
}
close $fh

puts "== reports written to $rep =="

# ---- pictures (need a window) ---------------------------------------------- #
set have_gui [expr {[info exists ::rdi::mode] && $::rdi::mode eq "gui"}]
if {!$have_gui} {
    puts "== batch mode: skipping schematic/device images. For those run:"
    puts "   vivado -mode gui -source rtl/report/hw_report.tcl -notrace -tclargs -dir $dir"
    exit 0
}

# Top-level schematic: the three blocks and their wiring, not 50k cells.
if {[catch {
    show_schematic [get_cells {u_top u_vio dbg_hub}]
    write_schematic -force -format pdf -orientation landscape $rep/schematic_top.pdf
} err]} { puts "schematic_top: $err" }

# One level down: the network's layers.
if {[catch {
    show_schematic [get_cells -quiet u_top/u_net/u_top/*]
    write_schematic -force -format pdf -orientation landscape $rep/schematic_network.pdf
} err]} { puts "schematic_network: $err" }

# Placed device with each major block highlighted in its own colour.
if {[catch {
    set colors {1 2 3 4 5 6 7 8}
    set k 0
    foreach blk {u_top/u_net/u_top/u_c1 u_top/u_net/u_top/u_b1 u_top/u_net/u_top/u_b2
                 u_top/u_net/u_top/u_b3 u_top/u_net/u_top/u_c2 u_top/u_net/u_top/u_tail
                 u_top/u_st u_vio} {
        set cells [get_cells -quiet -hier -filter "NAME =~ $blk/*"]
        if {[llength $cells]} {
            highlight_objects -color_index [lindex $colors $k] $cells
        }
        incr k
    }
    # write_device_image is not supported for spartan7 (Vivado 2026.1) -- take
    # the Device view by hand: File > Export > Export Image.
    write_device_image -force $rep/device_placed.png
} err]} { puts "device image: $err" }

puts "== pictures written to $rep =="
if {!$keep} { exit 0 }
