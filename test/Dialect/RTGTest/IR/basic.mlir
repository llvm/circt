// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: @cpus
// CHECK-SAME: !rtgtest.cpu
rtg.target @cpus : !rtg.dict<cpu: !rtgtest.cpu> {
  // CHECK: %0 = rtgtest.cpu_decl 0
  %0 = rtgtest.cpu_decl 0
  rtg.yield %0 : !rtgtest.cpu
}
