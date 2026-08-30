// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialized value.
// UNSUPPORTED: valgrind

// Slang models a labeled statement as a sequential block wrapped around the
// statement itself. A labeled concurrent assertion must still be emitted
// directly into the module body, the same way an unlabeled one is, instead of
// being buried in an `always` procedure that no later pass can dissolve.

// CHECK-LABEL: moore.module @Unlabeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert
module Unlabeled(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @Labeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert
module Labeled(input logic clk, input logic a, input logic b);
  my_assert: assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @LabeledAssume
// CHECK-NOT: moore.procedure
// CHECK: verif.assume
module LabeledAssume(input logic clk, input logic a, input logic b);
  my_assume: assume property (@(posedge clk) a |-> b);
endmodule

// Several labeled assertions in one module all land in the module body.
// CHECK-LABEL: moore.module @ManyLabeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert
// CHECK: verif.assert
// CHECK: verif.assert
module ManyLabeled(input logic clk, input logic a, input logic b);
  first:  assert property (@(posedge clk) a |-> b);
  second: assert property (@(posedge clk) b |-> a);
  third:  assert property (@(posedge clk) a);
endmodule

// A procedure that is not a concurrent assertion keeps its procedure.
// CHECK-LABEL: moore.module @OrdinaryProcedure
// CHECK: moore.procedure always_comb
module OrdinaryProcedure(input logic a, output logic b);
  always_comb begin
    b = a;
  end
endmodule
