// RUN: circt-test -l %s | FileCheck %s --check-prefix=CHECK-ALL
// RUN: circt-test -l --only-formal %s | FileCheck %s --check-prefix=CHECK-FORMAL
// RUN: circt-test -l --only-sim %s | FileCheck %s --check-prefix=CHECK-SIM

// CHECK-ALL: TestA
// CHECK-FORMAL: TestA
// CHECK-SIM-NOT: TestA
verif.formal @TestA {} {}

// CHECK-ALL: TestB
// CHECK-FORMAL-NOT: TestB
// CHECK-SIM: TestB
verif.simulation @TestB {} {
^bb0(%clock: !seq.clock, %init: i1):
  %0 = hw.constant true
  verif.yield %0, %0 : i1, i1
}
