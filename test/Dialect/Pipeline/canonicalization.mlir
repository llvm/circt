// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: hw.module @testNoDCE
// CHECK:   %[[DONE:.*]] = pipeline.scheduled
// CHECK:   hw.output
// CHECK: }
hw.module @testNoDCE(in %arg0: i1, in %clk: !seq.clock, in %rst: i1) {
  %done = pipeline.scheduled(%a0 : i1 = %arg0) clock(%clk) reset(%rst) go(%arg0) entryEn(%s0_enable) -> () {
    pipeline.return
  }
}
