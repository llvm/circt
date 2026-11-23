// RUN: circt-verilog %s | circt-opt --hw-flatten-modules --sroa --llhd-sig2reg --canonicalize --llhd-undriven-to-symbolic | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: hw.module @UndrivenVariable(
module UndrivenVariable(input logic sel, input logic [31:0] a,
                        output logic [31:0] y);
  // CHECK-NOT: llhd.
  // CHECK: %free = verif.symbolic_value : i32
  // CHECK-NOT: llhd.
  // CHECK: [[MUX:%.+]] = comb.mux %sel, %a, %free : i32
  // CHECK: hw.output [[MUX]] : i32
  logic [31:0] free; // Undriven
  assign y = sel ? a : free;
endmodule

// An undriven output port of a submodule becomes a free variable in the parent
// once the hierarchy is flattened.
module UndrivenChild(output logic [7:0] o);
endmodule

// CHECK-LABEL: hw.module @UndrivenOutput(
module UndrivenOutput(output logic [7:0] y);
  // CHECK-NOT: llhd.
  // CHECK: [[SYM:%.+]] = verif.symbolic_value name "c/o" : i8
  // CHECK-NOT: llhd.
  // CHECK: hw.output [[SYM]] : i8
  UndrivenChild c(.o(y));
endmodule
