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
// CHECK-NOT: label
module Unlabeled(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @Labeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert {{%.+}} label "my_assert"
module Labeled(input logic clk, input logic a, input logic b);
  my_assert: assert property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @LabeledAssume
// CHECK-NOT: moore.procedure
// CHECK: verif.assume {{%.+}} label "my_assume"
module LabeledAssume(input logic clk, input logic a, input logic b);
  my_assume: assume property (@(posedge clk) a |-> b);
endmodule

// CHECK-LABEL: moore.module @LabeledCover
// CHECK-NOT: moore.procedure
// CHECK: verif.cover {{%.+}} label "my_cover"
module LabeledCover(input logic clk, input logic a, input logic b);
  my_cover: cover property (@(posedge clk) a && b);
endmodule

// Several labeled assertions in one module all land in the module body.
// CHECK-LABEL: moore.module @ManyLabeled
// CHECK-NOT: moore.procedure
// CHECK: verif.assert {{%.+}} label "first"
// CHECK: verif.assert {{%.+}} label "second"
// CHECK: verif.assert {{%.+}} label "third"
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

// Slang wraps a module-level deferred immediate assertion in the same synthetic
// `always` procedure. Check that this isn't stripped.
// CHECK-LABEL: moore.module @DeferredImmediate
// CHECK: moore.procedure always {
// CHECK: moore.assert final
// CHECK: moore.procedure always {
// CHECK: moore.assume final
// CHECK: moore.procedure always {
// CHECK: moore.cover final
module DeferredImmediate(input logic a, input logic b);
  assert final (a);
  assume final (b);
  cover final (a && b);
endmodule

// Labeled concurrent assertions with `disable iff` keep both the enable and
// the label.
// CHECK-LABEL: moore.module @LabeledDisableIff
// CHECK: verif.assert {{%.+}} if {{%.+}} label "guarded"
module LabeledDisableIff(input logic clk, input logic rst, input logic a);
  guarded: assert property (@(posedge clk) disable iff (rst) a);
endmodule

// Labeled module-level deferred immediate assertions keep their label.
// CHECK-LABEL: moore.module @LabeledDeferredImmediate
// CHECK: moore.assert final {{%.+}} label "imm_assert"
// CHECK: moore.assume final {{%.+}} label "imm_assume"
// CHECK: moore.cover final {{%.+}} label "imm_cover"
module LabeledDeferredImmediate(input logic a, input logic b);
  imm_assert: assert final (a);
  imm_assume: assume final (b);
  imm_cover: cover final (a && b);
endmodule

// Labeled immediate assertions inside a procedure keep their label; unlabeled
// ones do not get one.
// CHECK-LABEL: moore.module @LabeledProcedural
// CHECK: moore.procedure always_comb
// CHECK: moore.assert immediate {{%.+}} label "proc_assert" :
// CHECK: moore.assert immediate {{%.+}}
// CHECK-NOT: label
// CHECK: moore.assume observed {{%.+}} label "proc_assume" :
module LabeledProcedural(input logic a, input logic b);
  always_comb begin
    proc_assert: assert (a);
    assert (b);
    proc_assume: assume #0 (a || b);
  end
endmodule
