// RUN: circt-opt %s | FileCheck %s

// TODO: replace this with `rtg.target` because ops implementing
// ContextResourceOpInterface are only allowed in such target operations.
// CHECK-LABEL: @cpus
rtg.sequence @cpus {
// CHECK: !rtgtest.cpu
^bb0(%arg0: !rtgtest.cpu):
  // CHECK: %0 = rtgtest.cpu_decl 0
  %0 = rtgtest.cpu_decl 0
}
