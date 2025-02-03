; RUN: firtool %s --format=fir --ir-fir    | circt-opt | FileCheck %s --check-prefix=MLIR
; RUN: firtool %s --format=fir --parse-only --annotation-file %s.anno.json,%s.anno.1.json | circt-opt | FileCheck %s --check-prefix=ANNOTATIONS
; RUN: firtool %s --format=fir --parse-only --annotation-file %s.anno.json --annotation-file %s.anno.1.json | circt-opt | FileCheck %s --check-prefix=ANNOTATIONS
; RUN: firtool %s --format=fir --ir-hw | circt-opt | FileCheck %s --check-prefix=MLIRLOWER
; RUN: firtool %s --format=fir -verilog | FileCheck %s --check-prefix=VERILOG
; RUN: firtool %s --format=fir --ir-verilog | circt-opt | FileCheck %s --check-prefix=VERILOG-IR
FIRRTL version 4.0.0

circuit test_mod : %[[{"class": "circt.testNT", "data": "a"}]]

; MLIR: firrtl.circuit "test_mod"

; ANNOTATIONS-LABEL: firrtl.circuit "test_mod"
; ANNOTATIONS-SAME: {class = "circt.testNT", data = "a"}
; ANNOTATIONS-SAME: {class = "circt.testNT", info = "a NoTargetAnnotation"}
; ANNOTATIONS-SAME: {class = "circt.test", info = "a CircuitTarget Annotation"}
; ANNOTATIONS-SAME: {class = "circt.test", info = "a CircuitName Annotation"}

; VERILOG-IR: sv.verbatim{{.*}}Standard header to adapt well known macros for register randomization
; VERILOG-IR: sv.macro.decl @ENABLE_INITIAL_REG_
; VERILOG-IR: sv.macro.decl @ENABLE_INITIAL_MEM_
; VERILOG-IR: sv.macro.decl @RANDOMIZE
; VERILOG-IR: sv.macro.decl @RANDOMIZE_DELAY
; VERILOG-IR: sv.macro.decl @RANDOM
; VERILOG-IR: sv.macro.decl @INIT_RANDOM
; VERILOG-IR: sv.macro.decl @INIT_RANDOM_PROLOG_

  type V3 = UInt<1>[3]
  type V2 = UInt<1>[2]

  public module test_mod :
    input clock : Clock
    input a: UInt<1>
    input b: UInt<2>
    output c: UInt<1>
    input vec: V3
    output out_implicitTrunc: UInt<1>
    output out_prettifyExample: UInt<1>
    output out_multibitMux: UInt<1>
    output out_mux2cell: V2

    inst cat of Cat
    connect cat.a, b
    connect cat.b, b
    connect cat.c, b

    inst implicitTrunc of ImplicitTrunc
    connect implicitTrunc.inp_1, a
    connect implicitTrunc.inp_2, asSInt(cat.d)

    inst prettifyExample of PrettifyExample
    connect prettifyExample.inp_1, cat.d
    connect prettifyExample.inp_2, cat.d
    connect prettifyExample.inp_3, cat.d

    inst flipFlop of FlipFlop
    connect flipFlop.clock, clock
    connect flipFlop.a_d, a

    connect c, flipFlop.a_q

    inst multibitMux of MultibitMux
    connect multibitMux.a, vec
    connect multibitMux.sel, b

    inst unusedPortsMod of UnusedPortsMod
    connect unusedPortsMod.in, a

    inst mux2cell of Mux2Cell
    connect mux2cell.cond, a
    connect mux2cell.low, a
    connect out_mux2cell, mux2cell.out

    ; These outputs exist to work around the aggressive removal of unused module
    ; ports.
    ;
    ; TODO: This test should be rewritten to not be so brittle around module
    ; port removal.
    connect out_implicitTrunc, or(orr(implicitTrunc.out1), orr(implicitTrunc.out2))
    connect out_prettifyExample, or(orr(prettifyExample.out1), orr(prettifyExample.out2))
    connect out_multibitMux, multibitMux.b


; MLIR-LABEL: firrtl.module @test_mod(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %c: !firrtl.uint<1>, in %vec_0: !firrtl.uint<1>, in %vec_1: !firrtl.uint<1>, in %vec_2: !firrtl.uint<1>, out %out_implicitTrunc: !firrtl.uint<1>, out %out_prettifyExample: !firrtl.uint<1>, out %out_multibitMux: !firrtl.uint<1>, out %out_mux2cell_0: !firrtl.uint<1>, out %out_mux2cell_1: !firrtl.uint<1>) {{.*}}{
; MLIR:         %cat_a, %cat_b, %cat_c, %cat_d = firrtl.instance cat @Cat(in a: !firrtl.uint<2>, in b: !firrtl.uint<2>, in c: !firrtl.uint<2>, out d: !firrtl.uint<6>)
; MLIR-NEXT:    firrtl.matchingconnect %cat_a, %b : !firrtl.uint<2>
; MLIR-NEXT:    firrtl.matchingconnect %cat_b, %b : !firrtl.uint<2>
; MLIR-NEXT:    firrtl.matchingconnect %cat_c, %b : !firrtl.uint<2>
; MLIR-NEXT:    %implicitTrunc_inp_1, %implicitTrunc_inp_2, %implicitTrunc_out1, %implicitTrunc_out2 = firrtl.instance implicitTrunc @ImplicitTrunc(in inp_1: !firrtl.uint<1>, in inp_2: !firrtl.sint<5>, out out1: !firrtl.sint<3>, out out2: !firrtl.sint<3>)
; MLIR-NEXT:    firrtl.matchingconnect %implicitTrunc_inp_1, %a : !firrtl.uint<1>
; MLIR-NEXT:    %0 = firrtl.asSInt %cat_d : (!firrtl.uint<6>) -> !firrtl.sint<6>
; MLIR-NEXT:    %1 = firrtl.bits %0 4 to 0 : (!firrtl.sint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    %2 = firrtl.asSInt %1 : (!firrtl.uint<5>) -> !firrtl.sint<5>
; MLIR-NEXT:    firrtl.matchingconnect %implicitTrunc_inp_2, %2 : !firrtl.sint<5>
; MLIR:         %prettifyExample_inp_1, %prettifyExample_inp_2, %prettifyExample_inp_3, %prettifyExample_out1, %prettifyExample_out2 = firrtl.instance prettifyExample @PrettifyExample(in inp_1: !firrtl.uint<5>, in inp_2: !firrtl.uint<5>, in inp_3: !firrtl.uint<5>, out out1: !firrtl.uint<10>, out out2: !firrtl.uint<10>)
; MLIR-NEXT:    %3 = firrtl.bits %cat_d 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    firrtl.matchingconnect %prettifyExample_inp_1, %3 : !firrtl.uint<5>
; MLIR-NEXT:    firrtl.matchingconnect %prettifyExample_inp_2, %3 : !firrtl.uint<5>
; MLIR-NEXT:    firrtl.matchingconnect %prettifyExample_inp_3, %3 : !firrtl.uint<5>
; MLIR-NEXT:    %flipFlop_clock, %flipFlop_a_d, %flipFlop_a_q = firrtl.instance flipFlop @FlipFlop(in clock: !firrtl.clock, in a_d: !firrtl.uint<1>, out a_q: !firrtl.uint<1>)
; MLIR-NEXT:    firrtl.matchingconnect %flipFlop_clock, %clock : !firrtl.clock
; MLIR-NEXT:    firrtl.matchingconnect %flipFlop_a_d, %a : !firrtl.uint<1>
; MLIR-NEXT:    firrtl.matchingconnect %c, %flipFlop_a_q : !firrtl.uint<1>
; MLIR-NEXT:    %multibitMux_a_0, %multibitMux_a_1, %multibitMux_a_2, %multibitMux_sel, %multibitMux_b = firrtl.instance multibitMux @MultibitMux(in a_0: !firrtl.uint<1>, in a_1: !firrtl.uint<1>, in a_2: !firrtl.uint<1>, in sel: !firrtl.uint<2>, out b: !firrtl.uint<1>)
; MLIR-NEXT:    firrtl.matchingconnect %multibitMux_a_0, %vec_0 : !firrtl.uint<1>
; MLIR-NEXT:    firrtl.matchingconnect %multibitMux_a_1, %vec_1 : !firrtl.uint<1>
; MLIR-NEXT:    firrtl.matchingconnect %multibitMux_a_2, %vec_2 : !firrtl.uint<1>
; MLIR-NEXT:    firrtl.matchingconnect %multibitMux_sel, %b : !firrtl.uint<2>

; ANNOTATIONS-LABEL: firrtl.module @test_mod
; ANNOTATIONS-SAME: {class = "circt.test", info = "a ModuleTarget Annotation"}
; ANNOTATIONS-SAME: {class = "circt.test", info = "a ModuleName Annotation"}

; VERILOG-LABEL: module test_mod(
; Check that a fir location is stripped.
; VERILOG-NOT:   .fir
; VERILOG-NEXT:    input        clock,
; VERILOG-NEXT:                 a,
; VERILOG-NEXT:    input  [1:0] b,
; VERILOG-NEXT:    output       c,
; VERILOG-NEXT:    input vec_0,
; VERILOG-NEXT:          vec_1,
; VERILOG-NEXT:          vec_2,
; VERILOG-NEXT:    output       out_implicitTrunc,
; VERILOG-NEXT:                 out_prettifyExample,
; VERILOG-NEXT:                 out_multibitMux,
; VERILOG-NEXT:                 out_mux2cell_0,
; VERILOG-NEXT:                 out_mux2cell_1
; VERILOG-NEXT:    );
; VERILOG-EMPTY:
; VERILOG-NEXT:    wire [9:0] _prettifyExample_out1;
; VERILOG-NEXT:    wire [9:0] _prettifyExample_out2;
; VERILOG-NEXT:    wire [2:0] _implicitTrunc_out1;
; VERILOG-NEXT:    wire [2:0] _implicitTrunc_out2;
; VERILOG-NEXT:    wire [5:0] _cat_d;
; VERILOG-NEXT:    Cat cat (
; VERILOG-NEXT:      .a (b),
; VERILOG-NEXT:      .b (b),
; VERILOG-NEXT:      .c (b),
; VERILOG-NEXT:      .d (_cat_d)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    ImplicitTrunc implicitTrunc (
; VERILOG-NEXT:      .inp_1 (a),
; VERILOG-NEXT:      .inp_2 (_cat_d[4:0]),
; VERILOG-NEXT:      .out1  (_implicitTrunc_out1),
; VERILOG-NEXT:      .out2  (_implicitTrunc_out2)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    PrettifyExample prettifyExample (
; VERILOG-NEXT:      .inp_1 (_cat_d[4:0]),
; VERILOG-NEXT:      .inp_2 (_cat_d[4:0]),
; VERILOG-NEXT:      .inp_3 (_cat_d[4:0]),
; VERILOG-NEXT:      .out1  (_prettifyExample_out1),
; VERILOG-NEXT:      .out2  (_prettifyExample_out2)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    FlipFlop flipFlop (
; VERILOG-NEXT:      .clock (clock),
; VERILOG-NEXT:      .a_d   (a),
; VERILOG-NEXT:      .a_q   (c)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    MultibitMux multibitMux (
; VERILOG-NEXT:     .a_0 (vec_0),
; VERILOG-NEXT:     .a_1 (vec_1),
; VERILOG-NEXT:     .a_2 (vec_2),
; VERILOG-NEXT:     .sel (b),
; VERILOG-NEXT:     .b   (out_multibitMux)
; VERILOG-NEXT:    );
; VERILOG:       endmodule

; Check that we canonicalize the HW output of lowering.

  module Cat :
    input a: UInt<2>
    input b: UInt<2>
    input c: UInt<2>
    output d: UInt<6>
    connect d, cat(cat(a, b), c)

; MLIRLOWER-LABEL: hw.module private @Cat(in %a : i2, in %b : i2, in %c : i2, out d : i6) {
; MLIRLOWER-NEXT:    %0 = comb.concat %a, %b, %c : i2, i2, i2
; MLIRLOWER-NEXT:    hw.output %0 : i6
; MLIRLOWER-NEXT:  }


; Check that implicit truncation is working.

  module ImplicitTrunc :
    input inp_1: UInt<1>
    input inp_2: SInt<5>
    output out1: SInt<3>
    output out2: SInt<3>
    connect out1, dshl(inp_2, inp_1)
    connect out2, inp_2

; MLIRLOWER-LABEL: hw.module private @ImplicitTrunc(in %inp_1 : i1, in %inp_2 : i5, out out1 : i3, out out2 : i3) {
; MLIRLOWER-NEXT:    %c0_i5 = hw.constant 0 : i5
; MLIRLOWER-NEXT:    %0 = comb.extract %inp_2 from 4 : (i5) -> i1
; MLIRLOWER-NEXT:    %1 = comb.concat %0, %inp_2 : i1, i5
; MLIRLOWER-NEXT:    %2 = comb.concat %c0_i5, %inp_1 : i5, i1
; MLIRLOWER-NEXT:    %3 = comb.shl %1, %2 : i6
; MLIRLOWER-NEXT:    %4 = comb.extract %3 from 0 : (i6) -> i3
; MLIRLOWER-NEXT:    %5 = comb.extract %inp_2 from 0 : (i5) -> i3
; MLIRLOWER-NEXT:    hw.output %4, %5 : i3, i3
; MLIRLOWER-NEXT:  }

; VERILOG-LABEL: module ImplicitTrunc(
; VERILOG-NEXT:   input        inp_1,
; VERILOG-NEXT:   input  [4:0] inp_2,
; VERILOG-NEXT:   output [2:0] out1,
; VERILOG-NEXT:                out2
; VERILOG-NEXT:   );
; VERILOG-EMPTY:
; VERILOG-NEXT:   wire [5:0] _GEN = {inp_2[4], inp_2} << inp_1;
; VERILOG-NEXT:   assign out1 = _GEN[2:0];
; VERILOG-NEXT:   assign out2 = inp_2[2:0];
; VERILOG-NEXT: endmodule


; Check that we prettify the IR before Verilog emission.

  module PrettifyExample :
    input inp_1: UInt<5>
    input inp_2: UInt<5>
    input inp_3: UInt<5>
    output out1: UInt<10>
    output out2: UInt<10>
    connect out1, cat(not(inp_1), inp_2)
    connect out2, cat(not(inp_1), inp_3)

; VERILOG-LABEL: module PrettifyExample(
; VERILOG:         assign out1 = {~inp_1, inp_2};
; VERILOG:         assign out2 = {~inp_1, inp_3};


; Check output of a simple flip-flop.

  module FlipFlop:
    input clock: Clock
    input a_d: UInt<1>
    output a_q: UInt<1>

    reg r: UInt<1>, clock

    connect r, a_d
    connect a_q, r

; VERILOG-LABEL: module FlipFlop(
; VERILOG-NEXT:    input  clock,
; VERILOG-NEXT:           a_d,
; VERILOG-NEXT:    output a_q
; VERILOG-NEXT:  );
; VERILOG:         always @(posedge clock)
; VERILOG-NEXT:      r <= a_d;
; VERILOG:         assign a_q = r;

  module MultibitMux :
    input a : UInt<1>[3]
    input sel : UInt<2>
    output b : UInt<1>
    connect b, a[sel]
; VERILOG-LABEL: module MultibitMux(
; VERILOG:  wire [3:0] [[T2:.*]] =
; VERILOG-SAME{LITERAL}: {{a_0}, {a_2}, {a_1}, {a_0}};
; VERILOG-NEXT:  assign b = [[T2]][sel];

  module UnusedPortsMod :
    input in : UInt<1>
    output out : UInt<1>
    invalidate out

  extmodule Val:
     output v: UInt<1>
  ; VERILOG-LABEL: Mux2Cell
  ; Make sure that mux annotations are emitted properly.
  ; VERILOG:      /* synopsys infer_mux_override */
  ; VERILOG-NEXT: assign [[OUT1:.+]] = mux2cell_in0 ? /* cadence map_to_mux */ mux2cell_in1 : mux2cell_in2;
  ; VERILOG:      /* synopsys infer_mux_override */
  ; VERILOG-NEXT: assign [[OUT2:.+]] =
  ; VERILOG-NEXT: mux2cell_in0_0 ? /* cadence map_to_mux */ mux2cell_in1_0 : mux2cell_in2_0;
  ; VERILOG: assign out_0 = [[OUT1]];
  ; VERILOG: assign out_1 = [[OUT2]];

  module Mux2Cell:
    input cond: UInt<1>
    input low: UInt<1>
    output out: UInt<1>[2]
    wire w: UInt<1>
    inst ext of Val
    connect w, ext.v
    connect out[0], intrinsic(circt_mux2cell : UInt, xor(cond, UInt<1>(1)), w, low)
    connect out[1], intrinsic(circt_mux2cell : UInt, xor(cond, UInt<1>(1)), ext.v, low)
