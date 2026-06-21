// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @ExtractPackedStruct
// CHECK-SAME: (%arg0: !hw.struct<a: i3, b: i2>) -> i2
func.func @ExtractPackedStruct(%arg0: !moore.struct<{a: i3, b: i2}>) -> !moore.i2 {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.struct<a: i3, b: i2>) -> i5
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[BITS]] from 1 : (i5) -> i2
  // CHECK: return [[EXTRACT]] : i2
  %0 = moore.extract %arg0 from 1 : !moore.struct<{a: i3, b: i2}> -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @DynExtractPackedStruct
// CHECK-SAME: (%arg0: !hw.struct<a: i3, b: i2>, %arg1: i3) -> i2
func.func @DynExtractPackedStruct(%arg0: !moore.struct<{a: i3, b: i2}>, %idx: !moore.i3) -> !moore.i2 {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.struct<a: i3, b: i2>) -> i5
  // CHECK: [[AMOUNT:%.+]] = comb.concat {{%.+}}, %arg1 : i2, i3
  // CHECK: [[SHIFTED:%.+]] = comb.shru [[BITS]], [[AMOUNT]] : i5
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[SHIFTED]] from 0 : (i5) -> i2
  // CHECK: return [[EXTRACT]] : i2
  %0 = moore.dyn_extract %arg0 from %idx : !moore.struct<{a: i3, b: i2}>, !moore.i3 -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @ExtractPackedUnion
// CHECK-SAME: (%arg0: !hw.union<a: i3, b: i5>) -> i2
func.func @ExtractPackedUnion(%arg0: !moore.union<{a: i3, b: i5}>) -> !moore.i2 {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.union<a: i3, b: i5>) -> i5
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[BITS]] from 1 : (i5) -> i2
  // CHECK: return [[EXTRACT]] : i2
  %0 = moore.extract %arg0 from 1 : !moore.union<{a: i3, b: i5}> -> !moore.i2
  return %0 : !moore.i2
}

// CHECK-LABEL: func.func @ExtractPackedStructToStruct
// CHECK-SAME: (%arg0: !hw.struct<a: i3, b: i2>) -> !hw.struct<b: i2>
func.func @ExtractPackedStructToStruct(%arg0: !moore.struct<{a: i3, b: i2}>) -> !moore.struct<{b: i2}> {
  // CHECK: [[BITS:%.+]] = hw.bitcast %arg0 : (!hw.struct<a: i3, b: i2>) -> i5
  // CHECK: [[EXTRACT:%.+]] = comb.extract [[BITS]] from 0 : (i5) -> i2
  // CHECK: [[RESULT:%.+]] = hw.bitcast [[EXTRACT]] : (i2) -> !hw.struct<b: i2>
  // CHECK: return [[RESULT]] : !hw.struct<b: i2>
  %0 = moore.extract %arg0 from 0 : !moore.struct<{a: i3, b: i2}> -> !moore.struct<{b: i2}>
  return %0 : !moore.struct<{b: i2}>
}

// CHECK-LABEL: func.func @ExtractIntToPackedStruct
// CHECK-SAME: (%arg0: i10) -> !hw.struct<a: i3, b: i2>
func.func @ExtractIntToPackedStruct(%arg0: !moore.l10) -> !moore.struct<{a: l3, b: l2}> {
  // CHECK: [[EXTRACT:%.+]] = comb.extract %arg0 from 5 : (i10) -> i5
  // CHECK: [[RESULT:%.+]] = hw.bitcast [[EXTRACT]] : (i5) -> !hw.struct<a: i3, b: i2>
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return [[RESULT]] : !hw.struct<a: i3, b: i2>
  %0 = moore.extract %arg0 from 5 : !moore.l10 -> !moore.struct<{a: l3, b: l2}>
  return %0 : !moore.struct<{a: l3, b: l2}>
}

// CHECK-LABEL: func.func @ExtractIntToPackedArray
// CHECK-SAME: (%arg0: i12) -> !hw.array<2xi3>
func.func @ExtractIntToPackedArray(%arg0: !moore.l12) -> !moore.array<2 x l3> {
  // CHECK: [[EXTRACT:%.+]] = comb.extract %arg0 from 6 : (i12) -> i6
  // CHECK: [[RESULT:%.+]] = hw.bitcast [[EXTRACT]] : (i6) -> !hw.array<2xi3>
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return [[RESULT]] : !hw.array<2xi3>
  %0 = moore.extract %arg0 from 6 : !moore.l12 -> !moore.array<2 x l3>
  return %0 : !moore.array<2 x l3>
}
