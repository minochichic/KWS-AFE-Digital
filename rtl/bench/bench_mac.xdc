# bench_mac -- three pins, same balls as rtl/constraints/kws_top.xdc uses them
# (clk = MAIN_CLOCK1 B6, rst_n = EXT16, out = EXT18). 50 MHz, the kit's clock.
create_clock -name sys_clk -period 20.000 [get_ports clk]
set_false_path -from [get_ports rst_n]
set_false_path -to   [get_ports out]

set_property PACKAGE_PIN B6   [get_ports clk]
set_property PACKAGE_PIN V16  [get_ports rst_n]
set_property PACKAGE_PIN AA16 [get_ports out]
set_property IOSTANDARD LVCMOS33 [get_ports {clk rst_n out}]
set_property PULLUP true [get_ports rst_n]
