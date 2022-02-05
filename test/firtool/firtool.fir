; RUN: firtool %s --format=fir --ir-fir    | circt-opt | FileCheck %s --check-prefix=MLIR
; RUN: firtool %s --format=fir --ir-fir --annotation-file %s.anno.json,%s.anno.1.json | circt-opt | FileCheck %s --check-prefix=ANNOTATIONS
; RUN: firtool %s --format=fir --ir-fir --annotation-file %s.anno.json --annotation-file %s.anno.1.json | circt-opt | FileCheck %s --check-prefix=ANNOTATIONS
; RUN: firtool %s --format=fir --ir-hw | circt-opt | FileCheck %s --check-prefix=MLIRLOWER
; RUN: firtool %s --format=fir -verilog |             FileCheck %s --check-prefix=VERILOG
; RUN: firtool %s --annotation-file %s.anno.json,%s.anno.1.json --parse-only | FileCheck %s --check-prefix=ANNOTATIONS
; RUN: firtool %s --omir-file %s.omir.anno.json --parse-only | FileCheck %s --check-prefix=OMIR
; RUN: firtool %s --omir-file %s.omir.anno.json --output-omir meta.omir.json --verilog | FileCheck %s --check-prefix=OMIROUT
; RUN: firtool %s --format=fir --ir-verilog | circt-opt | FileCheck %s --check-prefix=VERILOG-IR


circuit test_mod : %[[{"a": "a"}]]

; MLIR: firrtl.circuit "test_mod"

; ANNOTATIONS-LABEL: firrtl.circuit "test_mod"
; ANNOTATIONS-SAME: a = "a"
; ANNOTATIONS-SAME: info = "a NoTargetAnnotation"
; ANNOTATIONS-SAME: info = "a CircuitTarget Annotation
; ANNOTATIONS-SAME: info = "a CircuitName Annotation"

; OMIR:       #loc0 = loc(fused["Foo.scala":64:64, "Bar.scala":128:128])
; OMIR-NEXT:  #loc1 = loc("Foo.scala":32:32)
; OMIR-LABEL: firrtl.circuit "test_mod"
; OMIR-SAME:  {class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
; OMIR-SAME:   nodes = [
; OMIR-SAME:     {fields = {
; OMIR-SAME:        stringVal = {
; OMIR-SAME:          index = 0 : i64,
; OMIR-SAME:          info = #loc0,
; OMIR-SAME:          value = "OMString:hello"}},
; OMIR-SAME:      id = "OMID:0",
; OMIR-SAME:      info = #loc1}]}


; OMIROUT-LABEL: FILE "meta.omir.json" -----
; OMIROUT:     "info": "@[Foo.scala 32:32]",
; OMIROUT-NEXT:     "id": "OMID:0",
; OMIROUT-NEXT:     "fields": [
; OMIROUT-NEXT:       {
; OMIROUT-NEXT:         "info": "@[Foo.scala 64:64 Bar.scala 128:128]",
; OMIROUT-NEXT:         "name": "stringVal",
; OMIROUT-NEXT:         "value": "OMString:hello"
; OMIROUT-NEXT:       }

; VERILOG-IR: sv.verbatim{{.*}}Standard header to adapt well known macros to our needs
; VERILOG-IR: sv.verbatim "`define INIT_RANDOM_PROLOG_

  module test_mod :
    input clock : Clock
    input a: UInt<1>
    input b: UInt<2>
    output c: UInt<1>
    input vec: UInt<1>[3]

    inst cat of Cat
    cat.a <= b
    cat.b <= b
    cat.c <= b

    inst implicitTrunc of ImplicitTrunc
    implicitTrunc.inp_1 <= a
    implicitTrunc.inp_2 <= asSInt(cat.d)

    inst prettifyExample of PrettifyExample
    prettifyExample.inp_1 <= cat.d
    prettifyExample.inp_2 <= cat.d
    prettifyExample.inp_3 <= cat.d

    inst flipFlop of FlipFlop
    flipFlop.clock <= clock
    flipFlop.a_d <= a

    c <= flipFlop.a_q

    inst multibitMux of MultibitMux
    multibitMux.a <= vec
    multibitMux.sel <= b


; MLIR-LABEL: firrtl.module @test_mod(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %c: !firrtl.uint<1>, in %vec_0: !firrtl.uint<1>, in %vec_1: !firrtl.uint<1>, in %vec_2: !firrtl.uint<1>) {
; MLIR-NEXT:    %cat_a, %cat_b, %cat_c, %cat_d = firrtl.instance cat @Cat(in a: !firrtl.uint<2>, in b: !firrtl.uint<2>, in c: !firrtl.uint<2>, out d: !firrtl.uint<6>)
; MLIR-NEXT:    firrtl.connect %cat_a, %b : !firrtl.uint<2>, !firrtl.uint<2>
; MLIR-NEXT:    firrtl.connect %cat_b, %b : !firrtl.uint<2>, !firrtl.uint<2>
; MLIR-NEXT:    firrtl.connect %cat_c, %b : !firrtl.uint<2>, !firrtl.uint<2>
; MLIR-NEXT:    %implicitTrunc_inp_1, %implicitTrunc_inp_2, %implicitTrunc_out1, %implicitTrunc_out2 = firrtl.instance implicitTrunc @ImplicitTrunc(in inp_1: !firrtl.uint<1>, in inp_2: !firrtl.sint<5>, out out1: !firrtl.sint<3>, out out2: !firrtl.sint<3>)
; MLIR-NEXT:    firrtl.connect %implicitTrunc_inp_1, %a : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    %0 = firrtl.asSInt %cat_d : (!firrtl.uint<6>) -> !firrtl.sint<6>
; MLIR-NEXT:    %1 = firrtl.bits %0 4 to 0 : (!firrtl.sint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    %2 = firrtl.asSInt %1 : (!firrtl.uint<5>) -> !firrtl.sint<5>
; MLIR-NEXT:    firrtl.connect %implicitTrunc_inp_2, %2 : !firrtl.sint<5>, !firrtl.sint<5>
; MLIR-NEXT:    %prettifyExample_inp_1, %prettifyExample_inp_2, %prettifyExample_inp_3, %prettifyExample_out1, %prettifyExample_out2 = firrtl.instance prettifyExample @PrettifyExample(in inp_1: !firrtl.uint<5>, in inp_2: !firrtl.uint<5>, in inp_3: !firrtl.uint<5>, out out1: !firrtl.uint<10>, out out2: !firrtl.uint<10>)
; MLIR-NEXT:    %3 = firrtl.bits %cat_d 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    firrtl.connect %prettifyExample_inp_1, %3 : !firrtl.uint<5>, !firrtl.uint<5>
; MLIR-NEXT:    %4 = firrtl.bits %cat_d 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    firrtl.connect %prettifyExample_inp_2, %4 : !firrtl.uint<5>, !firrtl.uint<5>
; MLIR-NEXT:    %5 = firrtl.bits %cat_d 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
; MLIR-NEXT:    firrtl.connect %prettifyExample_inp_3, %5 : !firrtl.uint<5>, !firrtl.uint<5>
; MLIR-NEXT:    %flipFlop_clock, %flipFlop_a_d, %flipFlop_a_q = firrtl.instance flipFlop @FlipFlop(in clock: !firrtl.clock, in a_d: !firrtl.uint<1>, out a_q: !firrtl.uint<1>)
; MLIR-NEXT:    firrtl.connect %flipFlop_clock, %clock : !firrtl.clock, !firrtl.clock
; MLIR-NEXT:    firrtl.connect %flipFlop_a_d, %a : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    firrtl.connect %c, %flipFlop_a_q : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    %multibitMux_a_0, %multibitMux_a_1, %multibitMux_a_2, %multibitMux_sel, %multibitMux_b = firrtl.instance multibitMux @MultibitMux(in a_0: !firrtl.uint<1>, in a_1: !firrtl.uint<1>, in a_2: !firrtl.uint<1>, in sel: !firrtl.uint<2>, out b: !firrtl.uint<1>)
; MLIR-NEXT:    firrtl.connect %multibitMux_a_0, %vec_0 : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    firrtl.connect %multibitMux_a_1, %vec_1 : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    firrtl.connect %multibitMux_a_2, %vec_2 : !firrtl.uint<1>, !firrtl.uint<1>
; MLIR-NEXT:    firrtl.connect %multibitMux_sel, %b : !firrtl.uint<2>, !firrtl.uint<2>
; MLIR-NEXT:  }

; ANNOTATIONS-LABEL: firrtl.module @test_mod
; ANNOTATIONS-SAME: info = "a ModuleTarget Annotation"
; ANNOTATIONS-SAME: info = "a ModuleName Annotation"

; VERILOG-LABEL: module test_mod(
; VERILOG-NEXT:    input        clock, a,
; VERILOG-NEXT:    input  [1:0] b,
; VERILOG-NEXT:    input vec_0, vec_1, vec_2,
; VERILOG-NEXT:    output       c);
; VERILOG-EMPTY:
; VERILOG-NEXT:    wire multibitMux_b;
; VERILOG-NEXT:    wire [9:0] prettifyExample_out1;
; VERILOG-NEXT:    wire [9:0] prettifyExample_out2;
; VERILOG-NEXT:    wire [2:0] implicitTrunc_out1;
; VERILOG-NEXT:    wire [2:0] implicitTrunc_out2;
; VERILOG-NEXT:    wire [5:0] cat_d;
; VERILOG-EMPTY:
; VERILOG-NEXT:    Cat cat (
; VERILOG-NEXT:      .a (b),
; VERILOG-NEXT:      .b (b),
; VERILOG-NEXT:      .c (b),
; VERILOG-NEXT:      .d (cat_d)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    ImplicitTrunc implicitTrunc (
; VERILOG-NEXT:      .inp_1 (a),
; VERILOG-NEXT:      .inp_2 (cat_d[4:0]),
; VERILOG-NEXT:      .out1  (implicitTrunc_out1),
; VERILOG-NEXT:      .out2  (implicitTrunc_out2)
; VERILOG-NEXT:    );
; VERILOG-NEXT:    PrettifyExample prettifyExample (
; VERILOG-NEXT:      .inp_1 (cat_d[4:0]),
; VERILOG-NEXT:      .inp_2 (cat_d[4:0]),
; VERILOG-NEXT:      .inp_3 (cat_d[4:0]),
; VERILOG-NEXT:      .out1  (prettifyExample_out1),
; VERILOG-NEXT:      .out2  (prettifyExample_out2)
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
; VERILOG-NEXT:     .b   (multibitMux_b)
; VERILOG-NEXT:    );
; VERILOG-NEXT:  endmodule

; Check that we canonicalize the HW output of lowering.

  module Cat :
    input a: UInt<2>
    input b: UInt<2>
    input c: UInt<2>
    output d: UInt<6>
    d <= cat(cat(a, b), c)

; MLIRLOWER-LABEL: hw.module @Cat(%a: i2, %b: i2, %c: i2) -> (d: i6) {
; MLIRLOWER-NEXT:    %0 = comb.concat %a, %b, %c : i2, i2, i2
; MLIRLOWER-NEXT:    hw.output %0 : i6
; MLIRLOWER-NEXT:  }


; Check that implicit truncation is working.

  module ImplicitTrunc :
    input inp_1: UInt<1>
    input inp_2: SInt<5>
    output out1: SInt<3>
    output out2: SInt<3>
    out1 <= dshl(inp_2, inp_1)
    out2 <= inp_2

; MLIRLOWER-LABEL: hw.module @ImplicitTrunc(%inp_1: i1, %inp_2: i5) -> (out1: i3, out2: i3) {
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
; VERILOG-NEXT:   output [2:0] out1, out2);
; VERILOG-EMPTY:
; VERILOG-NEXT:   wire [5:0] _T = {inp_2[4], inp_2} << inp_1;
; VERILOG-NEXT:   assign out1 = _T[2:0];
; VERILOG-NEXT:   assign out2 = inp_2[2:0];
; VERILOG-NEXT: endmodule


; Check that we prettify the IR before Verilog emission.

  module PrettifyExample :
    input inp_1: UInt<5>
    input inp_2: UInt<5>
    input inp_3: UInt<5>
    output out1: UInt<10>
    output out2: UInt<10>
    out1 <= cat(not(inp_1), inp_2)
    out2 <= cat(not(inp_1), inp_3)

; VERILOG-LABEL: module PrettifyExample(
; VERILOG:         assign out1 = {~inp_1, inp_2};
; VERILOG:         assign out2 = {~inp_1, inp_3};


; Check output of a simple flip-flop.

  module FlipFlop:
    input clock: Clock
    input a_d: UInt<1>
    output a_q: UInt<1>

    reg r: UInt<1>, clock

    r <= a_d
    a_q <= r

; VERILOG-LABEL: module FlipFlop(
; VERILOG-NEXT:    input clock, a_d,
; VERILOG-NEXT:    output a_q);
; VERILOG:         always @(posedge clock)
; VERILOG-NEXT:      r <= a_d;
; VERILOG-NEXT:    assign a_q = r;

  module MultibitMux :
    input a : UInt<1>[3]
    input sel : UInt<2>
    output b : UInt<1>
    b <= a[sel]
; VERILOG-LABEL: module MultibitMux(
; VERILOG-CHECK: wire [2:0] [[T:.*]] = {{a_2}, {a_1}, {a_0}};
; VERILOG-CHECK-NEXT: assign b = [[T]][sel];