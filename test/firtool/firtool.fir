; RUN: firtool %s --format=fir -mlir    | circt-opt | FileCheck %s --check-prefix=MLIR
; RUN: firtool %s --format=fir -verilog |             FileCheck %s --check-prefix=VERILOG
; RUN: firtool %s --format=fir -mlir -lower-to-rtl | circt-opt | FileCheck %s --check-prefix=MLIRLOWER

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

  module cat :
    input a: UInt<2>
    input b: UInt<2>
    input c: UInt<2>
    output d: UInt<6>
    d <= cat(cat(a, b), c)

; MLIR: firrtl.circuit "test_mod" {

; MLIR-LABEL: firrtl.module @test_mod(%a: !firrtl.uint<1>, %b: !firrtl.flip<uint<1>>) {
; MLIR-NEXT:    firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
; MLIR-NEXT:  }

; VERILOG-LABEL: module test_mod(
; VERILOG-NEXT :   input  a,
; VERILOG-NEXT :   output b);
; VERILOG-NEXT :   assign b = a;
; VERILOG-NEXT : endmodule

; MLIRLOWER: module attributes {firrtl.mainModule = "test_mod"} {
; MLIRLOWER:   rtl.module @test_mod(%a: i1) -> (%b: i1) {
; MLIRLOWER:     rtl.output %a : i1
; MLIRLOWER:   }

; Check that we canonicalize the RTL output of lowering.
;
; MLIRLOWER: rtl.module @cat(%a: i2, %b: i2, %c: i2) -> (%d: i6) {
; MLIRLOWER:   %0 = rtl.concat %a, %b, %c : (i2, i2, i2) -> i6
; MLIRLOWER:   rtl.output %0 : i6
; MLIRLOWER: }

; MLIRLOWER: }

  module implicitTrunc :
    input inp_1: UInt<1>
    input inp_2: SInt<5>
    output out1: SInt<3>
    output out2: SInt<3>
    out1 <= dshl(inp_2, inp_1)
    out2 <= inp_2

; VERILOG-LABEL:   module implicitTrunc(
; VERILOG-NEXT :   input        inp_1,
; VERILOG-NEXT :   input  [4:0] inp_2,
; VERILOG-NEXT :   output [2:0] tmp9);
; VERILOG-NEXT :   assign out1 = $signed(inp_2) << inp_1;
; VERILOG-NEXT :   assign out2 = inp_2;
; VERILOG-NEXT : endmodule
