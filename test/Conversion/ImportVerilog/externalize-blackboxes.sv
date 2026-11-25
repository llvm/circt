// RUN: circt-verilog --ir-moore --top=Top %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-verilog --ir-moore --externalize-blackboxes --top=Top --top=TopBlackbox %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-hw --externalize-blackboxes --top=Top --top=TopBlackbox %s | FileCheck %s --check-prefix=HW
// RUN: circt-verilog --externalize-blackboxes --top=Top %s | circt-opt --export-verilog -o /dev/null | FileCheck %s --check-prefix=VERILOG
// REQUIRES: slang

// Helper referenced only from inside a blackbox body; skipped completely when
// `--externalize-blackboxes` is enabled.
module dead_helper(input logic [7:0] a, output logic [7:0] y);
  assign y = ~a;
endmodule

// DEFAULT: moore.module private @dead_helper
// MOORE-NOT: @dead_helper
// HW-NOT: @dead_helper
// VERILOG-NOT: module dead_helper

// Top-level blackbox remains public when externalized.
(* blackbox *)
module TopBlackbox(input logic [3:0] d, output logic [3:0] q);
  assign q = ~d;
endmodule

// MOORE-DAG: moore.module.extern @TopBlackbox(in %d : !moore.l4, out q : !moore.l4) attributes {sv.attributes = [#sv.attribute<"blackbox">]}
// HW-DAG: hw.module.extern @TopBlackbox(in %d : i4, out q : i4) attributes {sv.attributes = [#sv.attribute<"blackbox">]}

// Parameterized blackbox instantiated with two different specializations, plus
// a localparam that must NOT be emitted as an overridable parameter.
(* blackbox *)
module param_sram #(
  parameter int ADDR_W = 4,
  parameter int DATA_W = 8,
  parameter bit [7:0] MASK = 8'hFF,
  parameter string MODE = "fast",
  parameter type T = logic [DATA_W-1:0],
  localparam int DEPTH = 1 << ADDR_W
)(
  input  logic [ADDR_W-1:0] addr,
  input  T                  wdata,
  output T                  rdata
);
  // In IEEE 1800-2017 §6.20.1, a body `parameter` inside a module with a `#()`
  // parameter port list behaves as a `localparam` and must not be emitted as an
  // overridable parameter.
  parameter int BODY_LOCAL = ADDR_W + 1;
  // Behavioral/helper contents inside the blackbox must not be lowered.
  logic [7:0] unused_q;
  dead_helper u_dead(.a(addr[0] ? 8'hAA : 8'h55), .y(unused_q));
  assign rdata = wdata;
endmodule

// Verilog-2001 body-style `parameter` (no `#(...)` header) IS overridable.
// Also ensure cross-module hierarchical references inside an externalized
// blackbox body do not leak synthetic ports onto its signature.
(* blackbox *)
module body_param_bb(input logic a, output logic y);
  parameter int SCALE = 2;
  assign y = a & Top.x;
endmodule

// Parameterized blackbox exercising signed/negative, >32-bit unsigned, real,
// shortreal, 4-state integer with unknowns, and unpacked array parameters.
(* blackbox = 1 *)
module rich_params #(
  parameter int NEG = -4,
  parameter logic [63:0] WIDE = 64'h8000_0000_0000_0001,
  parameter real RVAL = 1.5,
  parameter shortreal SRVAL = 2.5,
  parameter logic [3:0] FOUR_STATE = 4'b10xz,
  parameter int ARR [0:1] = '{10, 20}
)(
  input  logic a,
  output logic y
);
  assign y = a;
endmodule

(* blackbox = "1" *)
module str_bb(input logic a, output logic y);
  assign y = ~a;
endmodule

// Explicitly disabled blackbox attributes stay regular `moore.module`s.
(* blackbox = 0 *)
module disabled_zero(input logic a, output logic y);
  assign y = a;
endmodule

(* blackbox = 1'b0 *)
module disabled_bit_zero(input logic a, output logic y);
  assign y = a;
endmodule

(* blackbox = "0" *)
module disabled_str_zero(input logic a, output logic y);
  assign y = a;
endmodule

(* blackbox = "" *)
module disabled_empty_str(input logic a, output logic y);
  assign y = a;
endmodule

(* blackbox = 1, blackbox = 0 *)
module disabled_last_wins(input logic a, output logic y);
  assign y = a;
endmodule

(* blackbox = 0, blackbox = 1 *)
module enabled_last_wins(input logic a, output logic y);
  assign y = ~a;
endmodule

// Non-blackbox module stays a regular `moore.module`.
module plain_sub(input logic a, output logic y);
  assign y = ~a;
endmodule

module Top(
  input  logic [3:0]  a0,
  input  logic [7:0]  d0,
  output logic [7:0]  q0,
  input  logic [5:0]  a1,
  input  logic [15:0] d1,
  output logic [15:0] q1,
  input  logic        x,
  output logic        y
);
  // DEFAULT: moore.module private @param_sram
  // DEFAULT: moore.module private @param_sram_0
  //
  // MOORE-DAG: moore.module.extern private @param_sram<ADDR_W: si32, DATA_W: si32, MASK: ui8, MODE: none, T: none>(in %addr : !moore.l4, in %wdata : !moore.l8, out rdata : !moore.l8) attributes {sv.attributes = [#sv.attribute<"blackbox">]}
  // MOORE-DAG: moore.module.extern private @param_sram_0<ADDR_W: si32, DATA_W: si32, MASK: ui8, MODE: none, T: none>(in %addr : !moore.l6, in %wdata : !moore.l16, out rdata : !moore.l16) attributes {sv.attributes = [#sv.attribute<"blackbox">], verilogName = "param_sram"}
  // MOORE-DAG: moore.module.extern private @body_param_bb<SCALE: si32>(in %a : !moore.l1, out y : !moore.l1) attributes {sv.attributes = [#sv.attribute<"blackbox">]}
  // MOORE-DAG: moore.module.extern private @rich_params<NEG: si32, WIDE: ui64, RVAL: f64, SRVAL: f32, FOUR_STATE: none, ARR: none>(in %a : !moore.l1, out y : !moore.l1) attributes {sv.attributes = [#sv.attribute<"blackbox" = "1">]}
  // MOORE-DAG: moore.module.extern private @str_bb(in %a : !moore.l1, out y : !moore.l1) attributes {sv.attributes = [#sv.attribute<"blackbox" = "\221\22">]}
  // MOORE-DAG: moore.module.extern private @enabled_last_wins(in %a : !moore.l1, out y : !moore.l1)
  // MOORE-DAG: moore.module private @disabled_zero
  // MOORE-DAG: moore.module private @disabled_bit_zero
  // MOORE-DAG: moore.module private @disabled_str_zero
  // MOORE-DAG: moore.module private @disabled_empty_str
  // MOORE-DAG: moore.module private @disabled_last_wins
  // MOORE-DAG: moore.module private @plain_sub
  // MOORE-NOT: DEPTH
  // MOORE-NOT: BODY_LOCAL
  //
  // HW-DAG: hw.module.extern private @param_sram<ADDR_W: si32, DATA_W: si32, MASK: ui8, MODE: none, T: none>(in %addr : i4, in %wdata : i8, out rdata : i8) attributes {sv.attributes = [#sv.attribute<"blackbox">]}
  // HW-DAG: hw.module.extern private @param_sram_0<ADDR_W: si32, DATA_W: si32, MASK: ui8, MODE: none, T: none>(in %addr : i6, in %wdata : i16, out rdata : i16) attributes {sv.attributes = [#sv.attribute<"blackbox">], verilogName = "param_sram"}
  // HW-DAG: hw.module.extern private @body_param_bb<SCALE: si32>(in %a : i1, out y : i1) attributes {sv.attributes = [#sv.attribute<"blackbox">]}
  // HW-DAG: hw.module.extern private @rich_params<NEG: si32, WIDE: ui64, RVAL: f64, SRVAL: f32, FOUR_STATE: none, ARR: none>(in %a : i1, out y : i1) attributes {sv.attributes = [#sv.attribute<"blackbox" = "1">]}
  // HW-DAG: hw.module.extern private @str_bb(in %a : i1, out y : i1)
  // HW-DAG: hw.module private @disabled_zero
  // HW-DAG: hw.module private @disabled_bit_zero
  // HW-DAG: hw.module private @disabled_str_zero
  // HW-DAG: hw.module private @plain_sub
  //
  // MOORE-LABEL: moore.module @Top
  // MOORE: moore.instance "u0" @param_sram<ADDR_W: si32 = 4, DATA_W: si32 = 8, MASK: ui8 = 170, MODE: none = "fast", T: none = #hw.param.verbatim<"logic[7:0]">>({{.*}}) {sv.attributes = [#sv.attribute<"keep_hierarchy">]}
  // MOORE: moore.instance "u1" @param_sram_0<ADDR_W: si32 = 6, DATA_W: si32 = 16, MASK: ui8 = 255, MODE: none = "slow", T: none = #hw.param.verbatim<"logic[15:0]">>(
  // MOORE: moore.instance "u_rich" @rich_params<NEG: si32 = -4, WIDE: ui64 = 9223372036854775809, RVAL: f64 = 1.500000e+00, SRVAL: f32 = 2.500000e+00, FOUR_STATE: none = #hw.param.verbatim<"4'b10xz">, ARR: none = #hw.param.verbatim<"'{10,20}">>(
  // MOORE: moore.instance "u_str" @str_bb(
  // MOORE: moore.instance "u_body_param" @body_param_bb<SCALE: si32 = 7>(
  //
  // HW-LABEL: hw.module @Top
  // HW: hw.instance "u0" @param_sram<ADDR_W: si32 = 4, DATA_W: si32 = 8, MASK: ui8 = 170, MODE: none = "fast", T: none = #hw.param.verbatim<"logic[7:0]">>({{.*}}) {sv.attributes = [#sv.attribute<"keep_hierarchy">]}
  // HW: hw.instance "u1" @param_sram_0<ADDR_W: si32 = 6, DATA_W: si32 = 16, MASK: ui8 = 255, MODE: none = "slow", T: none = #hw.param.verbatim<"logic[15:0]">>(
  // HW: hw.instance "u_rich" @rich_params<NEG: si32 = -4, WIDE: ui64 = 9223372036854775809, RVAL: f64 = 1.500000e+00, SRVAL: f32 = 2.500000e+00, FOUR_STATE: none = #hw.param.verbatim<"4'b10xz">, ARR: none = #hw.param.verbatim<"'{10,20}">>(
  // HW: hw.instance "u_str" @str_bb(
  // HW: hw.instance "u_body_param" @body_param_bb<SCALE: si32 = 7>(
  //
  // VERILOG-NOT: module param_sram
  // VERILOG-NOT: module rich_params
  // VERILOG-NOT: module str_bb
  // VERILOG-LABEL: module Top(
  // VERILOG: (* keep_hierarchy *)
  // VERILOG: param_sram #(
  // VERILOG:   .ADDR_W(4),
  // VERILOG:   .DATA_W(8),
  // VERILOG:   .MASK(170),
  // VERILOG:   .MODE("fast"),
  // VERILOG:   .T(logic[7:0])
  // VERILOG: ) u0 (
  // VERILOG: param_sram #(
  // VERILOG:   .ADDR_W(6),
  // VERILOG:   .DATA_W(16),
  // VERILOG:   .MASK(255),
  // VERILOG:   .MODE("slow"),
  // VERILOG:   .T(logic[15:0])
  // VERILOG: ) u1 (
  // VERILOG: rich_params #(
  // VERILOG:   .NEG(-4),
  // VERILOG:   .WIDE(64'd9223372036854775809),
  // VERILOG:   .RVAL(1.5
  // VERILOG:   .SRVAL(2.5
  // VERILOG:   .FOUR_STATE(4'b10xz),
  // VERILOG:   .ARR('{10,20})
  // VERILOG: ) u_rich (
  // VERILOG: str_bb u_str (
  // VERILOG: body_param_bb #(
  // VERILOG:   .SCALE(7)
  // VERILOG: ) u_body_param (
  // VERILOG-NOT: param_sram_0
  (* keep_hierarchy *)
  param_sram #(.ADDR_W(4), .DATA_W(8), .MASK(8'hAA), .MODE("fast")) u0 (
    .addr(a0), .wdata(d0), .rdata(q0)
  );
  param_sram #(.ADDR_W(6), .DATA_W(16), .MODE("slow")) u1 (
    .addr(a1), .wdata(d1), .rdata(q1)
  );

  logic m0, m1, m2, m3, m4, m5, m6, m7, m8;
  rich_params u_rich(.a(x), .y(m0));
  str_bb u_str(.a(m0), .y(m1));
  body_param_bb #(.SCALE(7)) u_body_param(.a(m1), .y(m2));
  disabled_zero u_d0(.a(m2), .y(m3));
  disabled_bit_zero u_d1(.a(m3), .y(m4));
  disabled_str_zero u_d2(.a(m4), .y(m5));
  disabled_empty_str u_d3(.a(m5), .y(m6));
  disabled_last_wins u_d4(.a(m6), .y(m7));
  enabled_last_wins u_e0(.a(m7), .y(m8));
  plain_sub u_p(.a(m8), .y(y));
endmodule
