# Continue implementation from a synthesized design checkpoint.
#
# Usage from the repository root:
#   vivado -mode batch -source rtl/implement_checkpoint.tcl -tclargs \
#     out/synth/xc7s75fgga484-1/post_synth.dcp

set ROOT [file normalize [file join [file dirname [info script]] ..]]
cd $ROOT

if {[llength $argv] < 1 || [llength $argv] > 3} {
    puts "usage: vivado -mode batch -source rtl/implement_checkpoint.tcl -tclargs <post_synth.dcp> ?out_dir? ?bit_name?"
    exit 2
}

set dcp [file normalize [lindex $argv 0]]
if {![file exists $dcp]} {
    puts "ERROR: synthesized checkpoint not found: $dcp"
    exit 1
}

if {[llength $argv] >= 2} {
    set out [file normalize [lindex $argv 1]]
} else {
    set out [file dirname $dcp]
}
set bit_name kws_stream_top.bit
if {[llength $argv] == 3} {
    set bit_name [lindex $argv 2]
}
file mkdir $out

puts "== opening synthesized checkpoint: $dcp =="
open_checkpoint $dcp

set clocks [get_clocks -quiet]
if {[llength $clocks] == 0} {
    puts "ERROR: checkpoint has no clocks; implementation timing would be meaningless."
    exit 1
}
puts "== clocks: $clocks =="

set pinned {}
foreach port [get_ports -quiet] {
    if {[get_property -quiet PACKAGE_PIN $port] ne ""} {
        lappend pinned $port
    }
}
puts "== ports with PACKAGE_PIN: [llength $pinned] / [llength [get_ports -quiet]] =="
if {[llength $pinned] == 0} {
    puts "ERROR: checkpoint has no pin constraints; refusing to create a bitstream."
    exit 1
}

opt_design
place_design
route_design

report_timing_summary -file [file join $out timing_impl.rpt]
report_utilization    -file [file join $out utilization_impl.rpt]
report_drc            -file [file join $out drc_impl.rpt]
write_checkpoint -force [file join $out post_route.dcp]

set bit [file join $out $bit_name]
write_bitstream -force $bit
puts "== bitstream: $bit =="
