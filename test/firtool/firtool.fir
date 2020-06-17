; RUN: firtool %s --format=fir -mlir    | circt-opt | FileCheck %s --check-prefix=MLIR
; RUN: firtool %s --format=fir -verilog |           FileCheck %s --check-prefix=VERILOG

circuit fir_test :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

; MLIR: firrtl.circuit "fir_test" {

; MLIR-LABEL: firrtl.module @test_mod(%a: !firrtl.uint<1>, %b: !firrtl.flip<uint<1>>) {
; MLIR-NEXT:    firrtl.connect %b, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
; MLIR-NEXT:  }

; VERILOG-LABEL: module test_mod(
; VERILOG-NEXT :   input  a,
; VERILOG-NEXT :   output b);
; VERILOG-NEXT :   assign b = a;
; VERILOG-NEXT : endmodule

