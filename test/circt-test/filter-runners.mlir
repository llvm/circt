// RUN: circt-test --dry-run %s | FileCheck %s

// Since we only have formal runners at the moment, only the formal tests should run.

// CHECK: TestA:
verif.formal @TestA {} {}

// CHECK: TestB:
verif.formal @TestB {} {}

// CHECK-NOT: TestC:
verif.simulation @TestC {} {
^bb0(%clock: !seq.clock, %init: i1):
  %0 = hw.constant true
  verif.yield %0, %0 : i1, i1
}

// CHECK: TestD:
verif.formal @TestD {} {}
