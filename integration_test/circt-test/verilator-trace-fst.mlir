// RUN: circt-test %s -d %t -r \verilator --trace=fst 2>&1 | FileCheck %s
// RUN: test -f %t/FSTTest/FSTTest.fst
// REQUIRES: verilator

// CHECK: 1 tests passed

verif.simulation @FSTTest {} {
^bb0(%clock: !seq.clock, %init: i1):
  %true = hw.constant true
  verif.yield %true, %true : i1, i1
}

