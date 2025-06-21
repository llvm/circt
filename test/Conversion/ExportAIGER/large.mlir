// RUN: circt-translate --export-aiger %s | circt-translate --import-aiger  | FileCheck %s

// CHECK-LABEL: hw.module @aiger_top
// Check first 2 and last 2 inputs
// CHECK: in %[[A0:[A-Z|a-z|0-9]+]] "a[0]" : i1
// CHECK: in %[[A1:[A-Z|a-z|0-9]+]] "a[1]" : i1
// CHECK: in %[[A298:[A-Z|a-z|0-9]+]] "a[298]" : i1
// CHECK: in %[[A299:[A-Z|a-z|0-9]+]] "a[299]" : i1
// CHECK: in %[[B0:[A-Z|a-z|0-9]+]] "b[0]" : i1
// CHECK: in %[[B1:[A-Z|a-z|0-9]+]] "b[1]" : i1
// CHECK: in %[[B298:[A-Z|a-z|0-9]+]] "b[298]" : i1
// CHECK: in %[[B299:[A-Z|a-z|0-9]+]] "b[299]" : i1
hw.module @mixed_logic(in %a: i300, in %b: i300, out x: i300, in %clk: !seq.clock) {
  %0 = aig.and_inv %a, not %b : i300
  // CHECK-NEXT: %[[AND_INV_0:.+]] = aig.and_inv not %[[B0]], %[[A0]] : i1
  // CHECK-NEXT: %[[AND_INV_1:.+]] = aig.and_inv not %[[B1]], %[[A1]] : i1
  // CHECK:      %[[AND_INV_298:.+]] = aig.and_inv not %[[B298]], %[[A298]] : i1
  // CHECK-NEXT: %[[AND_INV_299:.+]] = aig.and_inv not %[[B299]], %[[A299]] : i1
  // CHECK-NEXT: hw.output %[[AND_INV_0]], %[[AND_INV_1]]
  // CHECK-SAME: %[[AND_INV_298]], %[[AND_INV_299]] : i1
  hw.output %0 : i300
}
