// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @DynExtractPackedArrayToInt
// CHECK-SAME: (%arg0: !hw.array<2xi3>, %arg1: i3) -> i2
func.func @DynExtractPackedArrayToInt(
    %arg0: !moore.array<2 x i3>,
    %idx: !moore.i3
) -> !moore.i2 {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[SHIFTED:%.+]] = comb.shru [[BITS]], {{%.+}} : i6
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[SHIFTED]] from 0 : (i6) -> i2
  // CHECK: return [[EXTRACT]] : i2
  %0 = moore.dyn_extract %arg0 from %idx : !moore.array<2 x i3>, !moore.i3 -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @DynExtractPackedArrayToStruct
// CHECK-SAME: (%arg0: !hw.array<2xi3>, %arg1: i3) -> !hw.struct<b: i2>
func.func @DynExtractPackedArrayToStruct(
    %arg0: !moore.array<2 x i3>,
    %idx: !moore.i3
) -> !moore.struct<{b: i2}> {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[SHIFTED:%.+]] = comb.shru [[BITS]], {{%.+}} : i6
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[SHIFTED]] from 0 : (i6) -> i2
  // CHECK: [[RESULT:%.+]] = hw.bitcast [[EXTRACT]] : (i2) -> !hw.struct<b: i2>
  // CHECK: return [[RESULT]] : !hw.struct<b: i2>
  %0 = moore.dyn_extract %arg0 from %idx : !moore.array<2 x i3>, !moore.i3 -> !moore.struct<{b: i2}>
  return %0 : !moore.struct<{b: i2}>
}

// CHECK-LABEL: func.func @DynExtractPackedArrayElement
// CHECK-SAME: (%arg0: !hw.array<2xi3>, %arg1: i3) -> i3
func.func @DynExtractPackedArrayElement(
    %arg0: !moore.array<2 x i3>,
    %idx: !moore.i3
) -> !moore.i3 {
  // CHECK: [[VALUE:%.+]] = hw.array_get %arg0{{\[}}{{%.+}}{{\]}} : !hw.array<2xi3>, i1
  // CHECK: return [[VALUE]] : i3
  %0 = moore.dyn_extract %arg0 from %idx : !moore.array<2 x i3>, !moore.i3 -> !moore.i3
  return %0 : !moore.i3
}
