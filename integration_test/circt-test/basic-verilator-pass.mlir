// RUN: circt-test %s -d %t -r \verilator 2>&1 | FileCheck %s
// REQUIRES: verilator

// CHECK: 1 tests passed

verif.simulation @Foo {} {
^bb0(%clock: !seq.clock, %init: i1):
  %true = hw.constant true
  verif.yield %true, %true : i1, i1
}
