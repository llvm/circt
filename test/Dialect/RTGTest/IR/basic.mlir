// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: @cpus
// CHECK-SAME: !rtgtest.cpu
rtg.target @cpus : !rtg.dict<cpu: !rtgtest.cpu> {
  // CHECK: rtgtest.cpu_decl 0
  %0 = rtgtest.cpu_decl 0
  rtg.yield %0 : !rtgtest.cpu
}

// CHECK: rtgtest.constant_test i32 {value = "str"}
%1 = rtgtest.constant_test i32 {value = "str"}
