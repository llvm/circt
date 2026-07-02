// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @ExtractPackedArrayToInt
// CHECK-SAME: (%arg0: !hw.array<2xi3>) -> i2
func.func @ExtractPackedArrayToInt(%arg0: !moore.array<2 x i3>) -> !moore.i2 {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[BITS]] from 1 : (i6) -> i2
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return [[EXTRACT]] : i2
  %0 = moore.extract %arg0 from 1 : !moore.array<2 x i3> -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @ExtractPackedArrayToStruct
// CHECK-SAME: (%arg0: !hw.array<2xi3>) -> !hw.struct<b: i2>
func.func @ExtractPackedArrayToStruct(
    %arg0: !moore.array<2 x i3>
) -> !moore.struct<{b: i2}> {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[BITS]] from 1 : (i6) -> i2
  // CHECK: [[RESULT:%.+]] = hw.bitcast [[EXTRACT]] : (i2) -> !hw.struct<b: i2>
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return [[RESULT]] : !hw.struct<b: i2>
  %0 = moore.extract %arg0 from 1 : !moore.array<2 x i3> -> !moore.struct<{b: i2}>
  return %0 : !moore.struct<{b: i2}>
}

// CHECK-LABEL: func.func @ExtractPackedArrayElement
// CHECK-SAME: (%arg0: !hw.array<2xi3>) -> i3
func.func @ExtractPackedArrayElement(%arg0: !moore.array<2 x i3>) -> !moore.i3 {
  // CHECK: [[VALUE:%.+]] = hw.array_get %arg0
  // CHECK: return [[VALUE]] : i3
  %0 = moore.extract %arg0 from 1 : !moore.array<2 x i3> -> !moore.i3
  return %0 : !moore.i3
}
