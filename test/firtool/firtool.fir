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
; MLIRLOWER:   %0 = comb.concat %a, %b, %c : (i2, i2, i2) -> i6
; MLIRLOWER:   rtl.output %0 : i6
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

; MLIRLOWER-LABEL:   rtl.module @implicitTrunc(%inp_1: i1, %inp_2: i5) -> (%out1: i3, %out2: i3) {
; MLIRLOWER-NEXT :     %c0_i5 = comb.constant(0 : i5) : i5
; MLIRLOWER-NEXT :     %.out1.output = sv.wire  : !rtl.inout<i3>
; MLIRLOWER-NEXT :     %.out2.output = sv.wire  : !rtl.inout<i3>
; MLIRLOWER-NEXT :     %0 = comb.sext %inp_2 : (i5) -> i6
; MLIRLOWER-NEXT :     %1 = comb.concat %c0_i5, %inp_1 : (i5, i1) -> i6
; MLIRLOWER-NEXT :     %2 = comb.shl %0, %1 : i6
; MLIRLOWER-NEXT :     %3 = comb.extract %2 from 0 : (i6) -> i3
; MLIRLOWER-NEXT :     sv.connect %.out1.output, %3 : i3
; MLIRLOWER-NEXT :     %4 = comb.extract %inp_2 from 0 : (i5) -> i3
; MLIRLOWER-NEXT :     sv.connect %.out2.output, %4 : i3
; MLIRLOWER-NEXT :     %5 = sv.read_inout %.out1.output : !rtl.inout<i3>
; MLIRLOWER-NEXT :     %6 = sv.read_inout %.out2.output : !rtl.inout<i3>
; MLIRLOWER-NEXT :     rtl.output %5, %6 : i3, i3
; MLIRLOWER-NEXT :   }

; MLIRLOWER: }
