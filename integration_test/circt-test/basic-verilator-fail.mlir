// RUN: not circt-test %s -d %t -r \verilator 2>&1 | FileCheck %s
// REQUIRES: verilator

// CHECK: test Foo ... FAILED
// CHECK: 1 of 1 tests FAILED; 0 passed

verif.simulation @Foo {} {
^bb0(%clock: !seq.clock, %init: i1):
  %false = hw.constant false
  %true = hw.constant true
  verif.yield %true, %false : i1, i1
}
