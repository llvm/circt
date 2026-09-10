// RUN: circt-test %s -d %t -r \verilator --trace=vcd 2>&1 | FileCheck %s
// RUN: test -f %t/VCDTest/VCDTest.vcd
// REQUIRES: verilator

// CHECK: 1 tests passed

verif.simulation @VCDTest {} {
^bb0(%clock: !seq.clock, %init: i1):
  %true = hw.constant true
  verif.yield %true, %true : i1, i1
}

