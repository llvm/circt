// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @ExtractRefPackedArrayBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.array<2xi3>>) -> !llhd.ref<i2>
func.func @ExtractRefPackedArrayBitSlice(%arg0: !moore.ref<!moore.array<2 x i3>>) -> !moore.ref<!moore.i2> {
  // CHECK: [[IDX:%.+]] = hw.constant false
  // CHECK: [[ELEMENT:%.+]] = llhd.sig.array_get %arg0{{\[}}[[IDX]]{{\]}} : <!hw.array<2xi3>>
  // CHECK: [[LOW:%.+]] = hw.constant 1 : i2
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[ELEMENT]] from [[LOW]] : <i3> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.extract_ref %arg0 from 1 : !moore.ref<!moore.array<2 x i3>> -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @ExtractRefPackedArrayCrossElementBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.array<2xi3>>) -> !llhd.ref<i2>
func.func @ExtractRefPackedArrayCrossElementBitSlice(%arg0: !moore.ref<!moore.array<2 x i3>>) -> !moore.ref<!moore.i2> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.array<2xi3>> to !llhd.ref<i6>
  // CHECK: [[LOW:%.+]] = hw.constant 2 : i3
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[CAST]] from [[LOW]] : <i6> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.extract_ref %arg0 from 2 : !moore.ref<!moore.array<2 x i3>> -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @ExtractRefPackedStructField
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>) -> !llhd.ref<i2>
func.func @ExtractRefPackedStructField(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>) -> !moore.ref<!moore.i2> {
  // CHECK: [[RESULT:%.+]] = llhd.sig.struct_extract %arg0["b"] : <!hw.struct<a: i3, b: i2>>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.extract_ref %arg0 from 0 : !moore.ref<!moore.struct<{a: i3, b: i2}>> -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @ExtractRefPackedStructFieldBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>) -> !llhd.ref<i1>
func.func @ExtractRefPackedStructFieldBitSlice(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>) -> !moore.ref<!moore.i1> {
  // CHECK: [[FIELD:%.+]] = llhd.sig.struct_extract %arg0["b"] : <!hw.struct<a: i3, b: i2>>
  // CHECK: [[LOW:%.+]] = hw.constant true
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[FIELD]] from [[LOW]] : <i2> -> <i1>
  // CHECK: return [[RESULT]] : !llhd.ref<i1>
  %0 = moore.extract_ref %arg0 from 1 : !moore.ref<!moore.struct<{a: i3, b: i2}>> -> !moore.ref<!moore.i1>
  return %0 : !moore.ref<!moore.i1>
}

// CHECK-LABEL: func.func @ExtractRefPackedStructCrossFieldBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>) -> !llhd.ref<i2>
func.func @ExtractRefPackedStructCrossFieldBitSlice(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>) -> !moore.ref<!moore.i2> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.struct<a: i3, b: i2>> to !llhd.ref<i5>
  // CHECK: [[LOW:%.+]] = hw.constant 1 : i3
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[CAST]] from [[LOW]] : <i5> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.extract_ref %arg0 from 1 : !moore.ref<!moore.struct<{a: i3, b: i2}>> -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @DynExtractRefPackedArrayBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.array<2xi3>>, %arg1: i3) -> !llhd.ref<i2>
func.func @DynExtractRefPackedArrayBitSlice(%arg0: !moore.ref<!moore.array<2 x i3>>, %idx: !moore.i3) -> !moore.ref<!moore.i2> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.array<2xi3>> to !llhd.ref<i6>
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[CAST]] from %arg1 : <i6> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.dyn_extract_ref %arg0 from %idx : !moore.ref<!moore.array<2 x i3>>, !moore.i3 -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @DynExtractRefPackedStructBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>, %arg1: i3) -> !llhd.ref<i2>
func.func @DynExtractRefPackedStructBitSlice(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>, %idx: !moore.i3) -> !moore.ref<!moore.i2> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.struct<a: i3, b: i2>> to !llhd.ref<i5>
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[CAST]] from %arg1 : <i5> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.dyn_extract_ref %arg0 from %idx : !moore.ref<!moore.struct<{a: i3, b: i2}>>, !moore.i3 -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}

// CHECK-LABEL: func.func @ExtractRefPackedStructToStruct
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>) -> !llhd.ref<!hw.struct<x: i1, y: i1>>
func.func @ExtractRefPackedStructToStruct(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>) -> !moore.ref<!moore.struct<{x: i1, y: i1}>> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.struct<a: i3, b: i2>> to !llhd.ref<i5>
  // CHECK: [[LOW:%.+]] = hw.constant 1 : i3
  // CHECK: [[EXTRACT:%.+]] = llhd.sig.extract [[CAST]] from [[LOW]] : <i5> -> <i2>
  // CHECK: [[RESULT:%.+]] = builtin.unrealized_conversion_cast [[EXTRACT]] : !llhd.ref<i2> to !llhd.ref<!hw.struct<x: i1, y: i1>>
  // CHECK: return [[RESULT]] : !llhd.ref<!hw.struct<x: i1, y: i1>>
  %0 = moore.extract_ref %arg0 from 1 : !moore.ref<!moore.struct<{a: i3, b: i2}>> -> !moore.ref<!moore.struct<{x: i1, y: i1}>>
  return %0 : !moore.ref<!moore.struct<{x: i1, y: i1}>>
}

// CHECK-LABEL: func.func @ExtractRefPackedArrayToStruct
// CHECK-SAME: (%arg0: !llhd.ref<!hw.array<2xi3>>) -> !llhd.ref<!hw.struct<x: i1, y: i1>>
func.func @ExtractRefPackedArrayToStruct(%arg0: !moore.ref<!moore.array<2 x i3>>) -> !moore.ref<!moore.struct<{x: i1, y: i1}>> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.array<2xi3>> to !llhd.ref<i6>
  // CHECK: [[LOW:%.+]] = hw.constant 2 : i3
  // CHECK: [[EXTRACT:%.+]] = llhd.sig.extract [[CAST]] from [[LOW]] : <i6> -> <i2>
  // CHECK: [[RESULT:%.+]] = builtin.unrealized_conversion_cast [[EXTRACT]] : !llhd.ref<i2> to !llhd.ref<!hw.struct<x: i1, y: i1>>
  // CHECK: return [[RESULT]] : !llhd.ref<!hw.struct<x: i1, y: i1>>
  %0 = moore.extract_ref %arg0 from 2 : !moore.ref<!moore.array<2 x i3>> -> !moore.ref<!moore.struct<{x: i1, y: i1}>>
  return %0 : !moore.ref<!moore.struct<{x: i1, y: i1}>>
}

// CHECK-LABEL: func.func @DynExtractRefPackedStructToStruct
// CHECK-SAME: (%arg0: !llhd.ref<!hw.struct<a: i3, b: i2>>, %arg1: i3) -> !llhd.ref<!hw.struct<x: i1, y: i1>>
func.func @DynExtractRefPackedStructToStruct(%arg0: !moore.ref<!moore.struct<{a: i3, b: i2}>>, %idx: !moore.i3) -> !moore.ref<!moore.struct<{x: i1, y: i1}>> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.struct<a: i3, b: i2>> to !llhd.ref<i5>
  // CHECK: [[EXTRACT:%.+]] = llhd.sig.extract [[CAST]] from %arg1 : <i5> -> <i2>
  // CHECK: [[RESULT:%.+]] = builtin.unrealized_conversion_cast [[EXTRACT]] : !llhd.ref<i2> to !llhd.ref<!hw.struct<x: i1, y: i1>>
  // CHECK: return [[RESULT]] : !llhd.ref<!hw.struct<x: i1, y: i1>>
  %0 = moore.dyn_extract_ref %arg0 from %idx : !moore.ref<!moore.struct<{a: i3, b: i2}>>, !moore.i3 -> !moore.ref<!moore.struct<{x: i1, y: i1}>>
  return %0 : !moore.ref<!moore.struct<{x: i1, y: i1}>>
}

// CHECK-LABEL: func.func @DynExtractRefPackedArrayToStruct
// CHECK-SAME: (%arg0: !llhd.ref<!hw.array<2xi3>>, %arg1: i3) -> !llhd.ref<!hw.struct<x: i1, y: i1>>
func.func @DynExtractRefPackedArrayToStruct(%arg0: !moore.ref<!moore.array<2 x i3>>, %idx: !moore.i3) -> !moore.ref<!moore.struct<{x: i1, y: i1}>> {
  // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !llhd.ref<!hw.array<2xi3>> to !llhd.ref<i6>
  // CHECK: [[EXTRACT:%.+]] = llhd.sig.extract [[CAST]] from %arg1 : <i6> -> <i2>
  // CHECK: [[RESULT:%.+]] = builtin.unrealized_conversion_cast [[EXTRACT]] : !llhd.ref<i2> to !llhd.ref<!hw.struct<x: i1, y: i1>>
  // CHECK: return [[RESULT]] : !llhd.ref<!hw.struct<x: i1, y: i1>>
  %0 = moore.dyn_extract_ref %arg0 from %idx : !moore.ref<!moore.array<2 x i3>>, !moore.i3 -> !moore.ref<!moore.struct<{x: i1, y: i1}>>
  return %0 : !moore.ref<!moore.struct<{x: i1, y: i1}>>
}

// CHECK-LABEL: func.func @ExtractRefPackedUnionBitSlice
// CHECK-SAME: (%arg0: !llhd.ref<!hw.union<bits: i4, parts: !hw.struct<hi: i2, lo: i2>>>) -> !llhd.ref<i2>
func.func @ExtractRefPackedUnionBitSlice(%arg0: !moore.ref<!moore.union<{bits: i4, parts: struct<{hi: i2, lo: i2}>}>>) -> !moore.ref<!moore.i2> {
  // CHECK: [[FIELD:%.+]] = llhd.sig.struct_extract %arg0["bits"] : <!hw.union<bits: i4, parts: !hw.struct<hi: i2, lo: i2>>>
  // CHECK: [[LOW:%.+]] = hw.constant 0 : i2
  // CHECK: [[RESULT:%.+]] = llhd.sig.extract [[FIELD]] from [[LOW]] : <i4> -> <i2>
  // CHECK: return [[RESULT]] : !llhd.ref<i2>
  %0 = moore.extract_ref %arg0 from 0 : !moore.ref<!moore.union<{bits: i4, parts: struct<{hi: i2, lo: i2}>}>> -> !moore.ref<!moore.i2>
  return %0 : !moore.ref<!moore.i2>
}
