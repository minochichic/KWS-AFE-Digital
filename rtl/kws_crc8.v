// CRC-8, polynomial 0x07, init 0x00, no reflection, no final xor
// (CRC-8/SMBUS). One byte per call, combinational.
//
// Chosen because the STM32's CRC peripheral does exactly this with a 7-bit
// polynomial setting and no reflection, so the firmware computes it in
// hardware and both ends agree by construction. The check value over the
// ASCII string "123456789" is 0xF4 -- tests/test_wake_link.py pins it.
//
// A module rather than a function in an include file: build.tcl's include
// path does not contain the repo root, so a shared `include` would resolve in
// the simulator and not in Vivado.

`timescale 1ns/1ps
`default_nettype none

module kws_crc8 (
    input  wire [7:0] crc,
    input  wire [7:0] data,
    output wire [7:0] next
);
    wire [7:0] x0 = crc ^ data;
    wire [7:0] x1 = x0[7] ? {x0[6:0], 1'b0} ^ 8'h07 : {x0[6:0], 1'b0};
    wire [7:0] x2 = x1[7] ? {x1[6:0], 1'b0} ^ 8'h07 : {x1[6:0], 1'b0};
    wire [7:0] x3 = x2[7] ? {x2[6:0], 1'b0} ^ 8'h07 : {x2[6:0], 1'b0};
    wire [7:0] x4 = x3[7] ? {x3[6:0], 1'b0} ^ 8'h07 : {x3[6:0], 1'b0};
    wire [7:0] x5 = x4[7] ? {x4[6:0], 1'b0} ^ 8'h07 : {x4[6:0], 1'b0};
    wire [7:0] x6 = x5[7] ? {x5[6:0], 1'b0} ^ 8'h07 : {x5[6:0], 1'b0};
    wire [7:0] x7 = x6[7] ? {x6[6:0], 1'b0} ^ 8'h07 : {x6[6:0], 1'b0};
    assign next   = x7[7] ? {x7[6:0], 1'b0} ^ 8'h07 : {x7[6:0], 1'b0};
endmodule

`default_nettype wire
