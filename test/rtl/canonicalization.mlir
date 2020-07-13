// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @extract_noop(%arg0: i3) -> i3 {
func @extract_noop(%arg0: i3) -> i3 {
  // CHECK-NEXT: return %arg0
  %x = rtl.extract %arg0 from 0 : (i3) -> i3
  return %x : i3
}

// CHECK-LABEL: func @extract_cstfold() -> i3 {
func @extract_cstfold() -> i3 {
  %c42_i12 = rtl.constant(42 : i12) : i12
  // CHECK-NEXT: rtl.constant(-3 : i3)
  %x = rtl.extract %c42_i12 from 3 : (i12) -> i3
  return %x : i3
}

// CHECK-LABEL: func @and_noop(%arg0: i11) -> i11 {
// CHECK-NEXT:    return %arg0
func @and_noop(%arg0: i11) -> i11 {
  %0 = rtl.and %arg0 : i11
  %1 = rtl.or  %0 : i11
  %2 = rtl.xor %1 : i11
  %3 = rtl.add %2 : i11
  %4 = rtl.mul %3 : i11
  return %4 : i11
}
