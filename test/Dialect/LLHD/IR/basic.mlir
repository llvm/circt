// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @basic
// CHECK-SAME: (in [[IN0:%.+]] : i32, out out0 : i32)
hw.module @basic(in %in0 : i32, out out0 : i32) {
  // CHECK: %{{.*}} = llhd.delay [[IN0]] by <0ns, 1d, 0e> : i32
  %0 = llhd.delay %in0 by <0ns, 1d, 0e> : i32
  hw.output %0 : i32
}
