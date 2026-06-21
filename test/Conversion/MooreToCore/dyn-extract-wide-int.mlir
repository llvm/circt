// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @DynExtractWideInt
// CHECK-SAME: (%arg0: i4, %arg1: i2) -> i8
func.func @DynExtractWideInt(%value: !moore.i4, %idx: !moore.i2) -> !moore.i8 {
  // CHECK: [[ZERO:%.*]] = hw.constant 0 : i2
  // CHECK: [[AMOUNT:%.*]] = comb.concat [[ZERO]], %arg1 : i2, i2
  // CHECK: [[SHIFTED:%.*]] = comb.shru %arg0, [[AMOUNT]] : i4
  // CHECK: [[PAD:%.*]] = hw.constant 0 : i4
  // CHECK: [[WIDE:%.*]] = comb.concat [[PAD]], [[SHIFTED]] : i4, i4
  // CHECK: return [[WIDE]] : i8
  %0 = moore.dyn_extract %value from %idx : !moore.i4, !moore.i2 -> !moore.i8
  return %0 : !moore.i8
}

// CHECK-LABEL: func.func @DynExtractNarrowInt
// CHECK-SAME: (%arg0: i4, %arg1: i2) -> i2
func.func @DynExtractNarrowInt(%value: !moore.i4, %idx: !moore.i2) -> !moore.i2 {
  // CHECK: [[ZERO:%.*]] = hw.constant 0 : i2
  // CHECK: [[AMOUNT:%.*]] = comb.concat [[ZERO]], %arg1 : i2, i2
  // CHECK: [[SHIFTED:%.*]] = comb.shru %arg0, [[AMOUNT]] : i4
  // CHECK: [[NARROW:%.*]] = comb.extract [[SHIFTED]] from 0 : (i4) -> i2
  // CHECK: return [[NARROW]] : i2
  %0 = moore.dyn_extract %value from %idx : !moore.i4, !moore.i2 -> !moore.i2
  return %0 : !moore.i2
}
