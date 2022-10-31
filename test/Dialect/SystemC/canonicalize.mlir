// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @removeEmptySensitivityList
systemc.module @removeEmptySensitivityList(%in: !systemc.in<i1>) {
  // CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    systemc.sensitive
    // CHECK-NEXT: systemc.sensitive %in : !systemc.in<i1>
    systemc.sensitive %in : !systemc.in<i1>
  // CHECK-NEXT: }
  }
}
