// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// A real-valued conditional with an unknown (x) condition folds to an f64
// constant attribute. `hw.param.value` results are restricted to HW value
// types, so the HW constant materializer must decline instead of building an
// op the verifier rejects (ivtest pr2453002.v:
// "'hw.param.value' op result #0 must be a known primitive element, but got
// 'f64'").

// CHECK-LABEL: hw.module @top
// CHECK-NOT: hw.param.value
module top;
  parameter udef = 1'bx;
  real rl3 = udef ? 6 : 6.0;
  real rl4 = udef ? 7 : 7;
  initial begin
    if (rl3 != 6.0 || rl4 != 7.0)
      $display("FAILED");
    else
      $display("PASSED");
  end
endmodule
