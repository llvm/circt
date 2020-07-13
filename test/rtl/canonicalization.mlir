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

// CHECK-LABEL: func @variadic_noop(%arg0: i11) -> i11 {
func @variadic_noop(%arg0: i11) -> i11 {
  %0 = rtl.and %arg0 : i11
  %1 = rtl.or  %0 : i11
  %2 = rtl.xor %1 : i11
  %3 = rtl.add %2 : i11
  %4 = rtl.mul %3 : i11
  // CHECK-NEXT:    return %arg0
  return %4 : i11
}

// CHECK-LABEL: func @and_annulment(%arg0: i11, %arg1: i11) -> i11 {
func @and_annulment(%arg0: i11, %arg1: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.and %arg0, %arg1, %c0_i11 : i11
  // CHECK-NEXT: rtl.constant(0 : i11)
  return %0 : i11
}

// CHECK-LABEL: func @or_annulment(%arg0: i11) -> i11 {
func @or_annulment(%arg0: i11) -> i11 {
  %c-1_i11 = rtl.constant(-1 : i11) : i11
  %0 = rtl.or %arg0, %arg0, %arg0, %c-1_i11 : i11
  // CHECK-NEXT: rtl.constant(-1 : i11)
  return %0 : i11
}

// CHECK-LABEL: func @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
func @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
  // CHECK-NEXT: %c0_i11 = rtl.constant(0 : i11) : i11
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.mul %arg0, %arg1, %c0_i11 : i11
  // CHECK-NEXT: %0 = rtl.or %arg2, %c0_i11
  %1 = rtl.or %c0_i11, %arg2: i11
  // CHECK-NEXT: return %0
  return %1 : i11
}
