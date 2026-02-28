// RUN: circt-opt --strip-contracts %s | FileCheck %s

// CHECK-LABEL: func @foo
func.func @foo(%arg0: i42) -> i42 {
  // CHECK-NOT: verif.contract
  %0 = verif.contract %arg0 : i42 {}
  // CHECK: return %arg0
  return %0 : i42
}

// CHECK-LABEL: hw.module @bar
hw.module @bar() {
  // CHECK-NOT: verif.contract
  %0 = verif.contract %1 : i42 {}
  %1 = verif.contract %0 : i42 {}
}

// CHECK-LABEL: hw.module @baz
hw.module @baz(out z: i42) {
  // CHECK: [[TMP:%.+]] = verif.contract [[TMP]]
  // CHECK: hw.output [[TMP]]
  %2 = verif.contract %3 : i42 {}
  %3 = verif.contract %2 : i42 {}
  hw.output %3 : i42
}
