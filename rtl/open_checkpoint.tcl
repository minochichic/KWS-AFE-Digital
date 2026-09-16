# Open one synthesized or routed checkpoint in the Vivado GUI.
#
# Usage:
#   vivado -mode gui -source rtl/open_checkpoint.tcl \
#     -tclargs out/synth/xc7s75fgga484-1/post_synth.dcp

if {[llength $argv] != 1} {
    puts "ERROR: expected one checkpoint path"
    puts "usage: vivado -mode gui -source rtl/open_checkpoint.tcl -tclargs <file.dcp>"
    return -code error
}

set checkpoint [file normalize [lindex $argv 0]]
if {![file exists $checkpoint]} {
    puts "ERROR: checkpoint not found: $checkpoint"
    return -code error
}

open_checkpoint $checkpoint
puts "== opened checkpoint: $checkpoint =="
puts "== try: report_utilization -hierarchical =="
puts "== try: report_timing_summary =="
