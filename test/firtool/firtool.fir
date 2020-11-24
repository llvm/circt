; RUN: firtool %s --format=fir -mlir    | circt-opt | FileCheck %s --check-prefix=MLIR
; RUN: firtool %s --format=fir -verilog |           FileCheck %s --check-prefix=VERILOG
; RUN: firtool %s --format=fir -mlir -lower-to-rtl | circt-opt | FileCheck %s --check-prefix=MLIRLOWER

circuit test_mod :
  module test_mod :
    input a: UInt<1>
    output b: UInt<1>
    b <= a

  module Queue_1 :
    input clock : Clock
    input reset : UInt<1>
    output io_deq_bits_opcode : UInt<3>
  
    mem ram_opcode : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 2
      read-latency => 0
      write-latency => 1
      reader => io_deq_bits_MPORT
      writer => MPORT
      read-under-write => undefined
    io_deq_bits_opcode <= ram_opcode.io_deq_bits_MPORT.data @[Decoupled.scala 242:15]


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
; MLIRLOWER: }
