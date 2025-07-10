// RUN: circt-opt %s --export-verilog | FileCheck %s
//
// Este teste verifica se a operação 'comb.reverse' é corretamente
// traduzida para o operador de streaming do SystemVerilog.

// CHECK-LABEL: module test_reverse(
// CHECK-NEXT:    input  [7:0] in,
// CHECK-NEXT:    output [7:0] out
// CHECK-NEXT:  );
// CHECK-EMPTY:
// CHECK-NEXT:    assign out = {<<{in}};
// CHECK-NEXT:  endmodule

hw.module @test_reverse(in %in: i8, out out: i8) {
  %reversed = comb.reverse %in : (i8) -> i8
  hw.output %reversed : i8
}
