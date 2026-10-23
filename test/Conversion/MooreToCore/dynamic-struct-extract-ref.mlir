// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// A 131-bit struct needs an 8-bit bit offset, not a 1-bit field index.
// CHECK-LABEL: func.func @struct_slice
// CHECK-SAME: %arg0: !llhd.ref<!hw.struct<meta: i3, data: i128>>, %arg1: i32
func.func @struct_slice(%arg0: !moore.ref<struct<{meta: l3, data: l128}>>, %arg1: !moore.i32) -> !moore.ref<l32> {
  // CHECK: comb.extract %arg1 from 8 : (i32) -> i24
  // CHECK: [[INDEX:%.+]] = comb.mux {{.*}} : i8
  // CHECK: [[SLICE:%.+]] = llhd.sig.extract %arg0 from [[INDEX]] : <!hw.struct<meta: i3, data: i128>> -> <i32>
  // CHECK: return [[SLICE]] : !llhd.ref<i32>
  %slice = moore.dyn_extract_ref %arg0 from %arg1 : <struct<{meta: l3, data: l128}>>, i32 -> <l32>
  return %slice : !moore.ref<l32>
}

// Nested structs and arrays retain their packed layout. A narrow offset is
// extended to the bit index width of the entire struct.
// CHECK-LABEL: func.func @nested_slice
func.func @nested_slice(%arg0: !moore.ref<struct<{hi: struct<{a: i4, b: i4}>, lo: array<2 x i4>}>>, %arg1: !moore.i2) -> !moore.ref<i6> {
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i2
  // CHECK: [[INDEX:%.+]] = comb.concat [[ZERO]], %arg1 : i2, i2
  // CHECK: [[SLICE:%.+]] = llhd.sig.extract %arg0 from [[INDEX]] : <!hw.struct<hi: !hw.struct<a: i4, b: i4>, lo: !hw.array<2xi4>>> -> <i6>
  // CHECK: return [[SLICE]] : !llhd.ref<i6>
  %slice = moore.dyn_extract_ref %arg0 from %arg1 : <struct<{hi: struct<{a: i4, b: i4}>, lo: array<2 x i4>}>>, i2 -> <i6>
  return %slice : !moore.ref<i6>
}
