// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// A statement label on a module-level concurrent assertion (`CHK: assert
// property (p);`) wraps the statement in a single-statement block, which
// used to defeat the module-scope conversion short-circuit: the labeled
// assertion converted inside a `moore.procedure always` while its unlabeled
// twin converted at module scope, so downstream consumers of module-scope
// assertions never saw the labeled one. A label must not change which
// lowering an assertion gets.

// CHECK-LABEL: moore.module @LabeledAssert
module LabeledAssert(input logic clk, input logic a, input logic b);
  // Unlabeled and labeled convert identically, at module scope.
  // CHECK: ltl.implication
  // CHECK: ltl.clock
  // CHECK: verif.assert
  // CHECK: ltl.implication
  // CHECK: ltl.clock
  // CHECK: verif.assert
  // CHECK-NOT: moore.procedure
  assert property (@(posedge clk) a |-> b);
  CHK: assert property (@(posedge clk) a |-> b);
endmodule

// A `disable iff` reset rides along unchanged.
// CHECK-LABEL: moore.module @LabeledAssertDisableIff
module LabeledAssertDisableIff(input logic clk, input logic rst,
                               input logic a, input logic b);
  // CHECK: ltl.implication
  // CHECK: ltl.clock
  // CHECK: verif.assert {{%.+}} if {{%.+}} : !ltl.property
  // CHECK-NOT: moore.procedure
  DCHK: assert property (@(posedge clk) disable iff (rst) a |-> b);
endmodule

// Default clocking is resolved the same way at module scope: the sampled
// value function picks up the default clock, and the assertion converts
// like its unlabeled twin.
// CHECK-LABEL: moore.module @LabeledAssertDefaultClocking
module LabeledAssertDefaultClocking(input logic clk, input logic [7:0] d);
  default clocking @(posedge clk); endclocking
  // CHECK: ltl.past
  // CHECK: verif.assert
  // CHECK-NOT: moore.procedure
  PCHK: assert property (d == $past(d));
endmodule

// Concurrent assertions written in an `initial` block have one-attempt
// semantics (IEEE 1800-2017 16.14.6) and must NOT be rerouted to a
// continuous module-scope assertion by the label look-through.
// CHECK-LABEL: moore.module @InitialNotRerouted
module InitialNotRerouted(input logic clk, input logic a);
  // CHECK: moore.procedure initial
  // CHECK: ltl.clock
  // CHECK: verif.assert
  initial begin
    LBL2: assert property (@(posedge clk) a);
  end
endmodule
